//! X11 output backend (SPEC §8, backend 3): synthesized key events via the
//! XTEST extension. Retraction is synthesized BackSpace.
//!
//! Keycodes are resolved against the *current* keyboard mapping, so this
//! respects the user's layout; characters the layout can't produce with at
//! most Shift fail loudly rather than typing the wrong glyph.

use std::collections::HashMap;

use anyhow::{Context as _, Result};
use x11rb::connection::Connection;
use x11rb::protocol::xproto::{ConnectionExt as _, KEY_PRESS_EVENT, KEY_RELEASE_EVENT};
use x11rb::protocol::xtest::ConnectionExt as _;
use x11rb::rust_connection::RustConnection;

use crate::{Caps, Injector};

const KEYSYM_BACKSPACE: u32 = 0xff08;
const KEYSYM_RETURN: u32 = 0xff0d;
const KEYSYM_SHIFT_L: u32 = 0xffe1;

fn keysym_of(c: char) -> u32 {
    match c {
        '\n' => KEYSYM_RETURN,
        '\u{8}' => KEYSYM_BACKSPACE,
        // Latin-1 keysyms are the codepoint; everything else uses the
        // Unicode keysym range.
        _ if (c as u32) < 0x100 => c as u32,
        _ => 0x0100_0000 + c as u32,
    }
}

pub struct XtestInjector {
    conn: RustConnection,
    root: u32,
    /// keysym -> (keycode, needs shift)
    keycodes: HashMap<u32, (u8, bool)>,
    shift: u8,
}

impl XtestInjector {
    pub fn new() -> Result<Self> {
        let (conn, screen_num) = x11rb::connect(None).context("connect to X display")?;
        conn.xtest_get_version(2, 2)
            .context("XTEST request")?
            .reply()
            .context("XTEST extension missing")?;
        let setup = conn.setup();
        let root = setup.roots[screen_num].root;
        let min = setup.min_keycode;
        let count = setup.max_keycode - min + 1;
        let mapping = conn
            .get_keyboard_mapping(min, count)?
            .reply()
            .context("get keyboard mapping")?;
        let per = mapping.keysyms_per_keycode as usize;
        let mut keycodes = HashMap::new();
        for (i, syms) in mapping.keysyms.chunks(per).enumerate() {
            let kc = min + i as u8;
            // Column 0 = plain, column 1 = shifted (group 1). First keycode
            // that produces a keysym wins; prefer unshifted.
            for (col, &ks) in syms.iter().take(2).enumerate() {
                if ks != 0 {
                    keycodes.entry(ks).or_insert((kc, col == 1));
                }
            }
        }
        let shift = keycodes
            .get(&KEYSYM_SHIFT_L)
            .map(|&(kc, _)| kc)
            .context("layout has no Shift_L")?;
        Ok(XtestInjector { conn, root, keycodes, shift })
    }

    fn tap(&self, c: char) -> Result<()> {
        let &(kc, shifted) = self
            .keycodes
            .get(&keysym_of(c))
            .with_context(|| format!("current layout cannot type {c:?}"))?;
        if shifted {
            self.conn
                .xtest_fake_input(KEY_PRESS_EVENT, self.shift, 0, self.root, 0, 0, 0)?;
        }
        self.conn
            .xtest_fake_input(KEY_PRESS_EVENT, kc, 0, self.root, 0, 0, 0)?;
        self.conn
            .xtest_fake_input(KEY_RELEASE_EVENT, kc, 0, self.root, 0, 0, 0)?;
        if shifted {
            self.conn
                .xtest_fake_input(KEY_RELEASE_EVENT, self.shift, 0, self.root, 0, 0, 0)?;
        }
        Ok(())
    }

    /// Push everything through and surface X errors as a Result.
    fn sync(&self) -> Result<()> {
        self.conn.get_input_focus()?.reply()?;
        Ok(())
    }
}

impl Injector for XtestInjector {
    fn commit(&mut self, text: &str) -> Result<()> {
        for c in text.chars() {
            self.tap(c)?;
        }
        self.sync()
    }

    fn retract(&mut self, text: &str) -> Result<()> {
        for _ in text.chars() {
            self.tap('\u{8}')?;
        }
        self.sync()
    }

    fn capabilities(&self) -> Caps {
        Caps { name: "x11-xtest", true_retraction: false }
    }
}

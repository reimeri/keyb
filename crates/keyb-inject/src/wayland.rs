//! Wayland output backends (SPEC §8, backends 1–2).
//!
//! One connection carries both protocols. Per commit, prefer
//! `zwp_input_method_v2` when the compositor has activated us for the focused
//! text field (real commit_string + delete_surrounding_text); otherwise fall
//! back to `zwp_virtual_keyboard_v1` key synthesis, which reaches focused
//! clients that speak no text-input protocol at all (terminals, X11 apps via
//! Xwayland, games).
//!
//! The virtual keyboard uploads a purpose-built xkb keymap in which every
//! alphabet character owns one keycode (the wtype trick), so injection is
//! layout-independent.

use std::io::Write;
use std::os::fd::{AsFd, FromRawFd, OwnedFd};
use std::time::Instant;

use anyhow::{bail, Context as _, Result};
use wayland_client::globals::{registry_queue_init, GlobalListContents};
use wayland_client::protocol::{wl_registry, wl_seat};
use wayland_client::{Connection, Dispatch, EventQueue, QueueHandle};
use wayland_protocols_misc::zwp_input_method_v2::client::{
    zwp_input_method_manager_v2::ZwpInputMethodManagerV2,
    zwp_input_method_v2::{self, ZwpInputMethodV2},
};
use wayland_protocols_misc::zwp_virtual_keyboard_v1::client::{
    zwp_virtual_keyboard_manager_v1::ZwpVirtualKeyboardManagerV1,
    zwp_virtual_keyboard_v1::ZwpVirtualKeyboardV1,
};

use crate::{Caps, Injector};

const KEY_PRESSED: u32 = 1;
const KEY_RELEASED: u32 = 0;

struct State {
    /// Input method activation, double-buffered per protocol: activate /
    /// deactivate arrive between `done` events.
    im_pending_active: bool,
    im_active: bool,
    /// Count of `done` events; `commit` must carry this as its serial.
    im_serial: u32,
    /// Another IM (an IME like fcitx) owns the seat; IM path is dead.
    im_unavailable: bool,
    /// The focused client reports surrounding text — only then is
    /// `delete_surrounding_text` retraction trustworthy (terminals never
    /// send it and silently drop deletions).
    im_surrounding: bool,
    /// When activation last flipped on: targets need a moment to finish
    /// their own IME setup before the first commit reliably lands.
    im_activated_at: Option<Instant>,
}

pub struct WaylandInjector {
    _conn: Connection,
    queue: EventQueue<State>,
    state: State,
    im: Option<ZwpInputMethodV2>,
    vk: Option<ZwpVirtualKeyboardV1>,
    /// Symbol -> evdev keycode in our uploaded keymap.
    chars: Vec<char>,
    epoch: Instant,
    /// Keep the keymap fd alive for the lifetime of the virtual keyboard.
    _keymap_fd: Option<OwnedFd>,
}

impl Dispatch<wl_registry::WlRegistry, GlobalListContents> for State {
    fn event(
        _: &mut State,
        _: &wl_registry::WlRegistry,
        _: wl_registry::Event,
        _: &GlobalListContents,
        _: &Connection,
        _: &QueueHandle<State>,
    ) {
    }
}

impl Dispatch<wl_seat::WlSeat, ()> for State {
    fn event(
        _: &mut State,
        _: &wl_seat::WlSeat,
        _: wl_seat::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<State>,
    ) {
    }
}

impl Dispatch<ZwpInputMethodManagerV2, ()> for State {
    fn event(
        _: &mut State,
        _: &ZwpInputMethodManagerV2,
        _: <ZwpInputMethodManagerV2 as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<State>,
    ) {
    }
}

impl Dispatch<ZwpVirtualKeyboardManagerV1, ()> for State {
    fn event(
        _: &mut State,
        _: &ZwpVirtualKeyboardManagerV1,
        _: <ZwpVirtualKeyboardManagerV1 as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<State>,
    ) {
    }
}

impl Dispatch<ZwpVirtualKeyboardV1, ()> for State {
    fn event(
        _: &mut State,
        _: &ZwpVirtualKeyboardV1,
        _: <ZwpVirtualKeyboardV1 as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<State>,
    ) {
    }
}

impl Dispatch<ZwpInputMethodV2, ()> for State {
    fn event(
        state: &mut State,
        _: &ZwpInputMethodV2,
        event: zwp_input_method_v2::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<State>,
    ) {
        match event {
            zwp_input_method_v2::Event::Activate => state.im_pending_active = true,
            zwp_input_method_v2::Event::Deactivate => {
                state.im_pending_active = false;
                state.im_surrounding = false;
            }
            zwp_input_method_v2::Event::SurroundingText { .. } => state.im_surrounding = true,
            zwp_input_method_v2::Event::Done => {
                if !state.im_active && state.im_pending_active {
                    state.im_activated_at = Some(Instant::now());
                }
                state.im_active = state.im_pending_active;
                state.im_serial = state.im_serial.wrapping_add(1);
            }
            zwp_input_method_v2::Event::Unavailable => state.im_unavailable = true,
            _ => {}
        }
    }
}

/// xkb keysym name for a symbol in the generated keymap. Unicode "U+XXXX"
/// names keep this layout-independent; control chars get their named syms.
fn keysym_name(c: char) -> String {
    match c {
        '\n' => "Return".into(),
        '\u{8}' => "BackSpace".into(),
        _ => format!("U{:04X}", c as u32),
    }
}

/// Minimal xkb_v1 keymap: symbol `i` of `chars` lives on evdev keycode
/// `i + 1` (xkb keycode `i + 9`).
fn build_keymap(chars: &[char]) -> String {
    let mut codes = String::new();
    let mut syms = String::new();
    for (i, &c) in chars.iter().enumerate() {
        let xkb = i + 9;
        codes.push_str(&format!("        <K{i}> = {xkb};\n"));
        syms.push_str(&format!("        key <K{i}> {{ [ {} ] }};\n", keysym_name(c)));
    }
    format!(
        "xkb_keymap {{\n\
         xkb_keycodes \"keyb\" {{\n\
         \x20       minimum = 8;\n\
         \x20       maximum = {max};\n\
         {codes}\
         }};\n\
         xkb_types \"keyb\" {{ include \"complete\" }};\n\
         xkb_compatibility \"keyb\" {{ include \"complete\" }};\n\
         xkb_symbols \"keyb\" {{\n\
         {syms}\
         }};\n\
         }};\n",
        max = chars.len() + 9,
    )
}

fn memfd_with(content: &str) -> Result<OwnedFd> {
    let fd = unsafe {
        libc::memfd_create(c"keyb-keymap".as_ptr(), libc::MFD_CLOEXEC)
    };
    if fd < 0 {
        bail!("memfd_create failed: {}", std::io::Error::last_os_error());
    }
    let mut f = unsafe { std::fs::File::from_raw_fd(fd) };
    f.write_all(content.as_bytes())?;
    f.write_all(&[0])?; // compositors read this as a C string
    Ok(f.into())
}

impl WaylandInjector {
    /// `chars` is the alphabet the virtual keyboard must be able to type.
    /// BackSpace is always included (retraction).
    pub fn new(chars: &[char]) -> Result<Self> {
        Self::with_options(chars, true)
    }

    /// Virtual-keyboard only — skips input-method-v2 even when the
    /// compositor offers it (testing, or a broken compositor IM).
    pub fn vk_only(chars: &[char]) -> Result<Self> {
        Self::with_options(chars, false)
    }

    fn with_options(chars: &[char], use_im: bool) -> Result<Self> {
        let conn = Connection::connect_to_env().context("connect to wayland display")?;
        let (globals, mut queue) = registry_queue_init::<State>(&conn)?;
        let qh = queue.handle();

        let seat: wl_seat::WlSeat = globals
            .bind(&qh, 1..=4, ())
            .context("compositor has no wl_seat")?;
        let im_manager: Option<ZwpInputMethodManagerV2> = if use_im {
            globals.bind(&qh, 1..=1, ()).ok()
        } else {
            None
        };
        let vk_manager: Option<ZwpVirtualKeyboardManagerV1> = globals.bind(&qh, 1..=1, ()).ok();
        if im_manager.is_none() && vk_manager.is_none() {
            bail!("compositor offers neither input-method-v2 nor virtual-keyboard-v1");
        }

        // '\u{8}' (BackSpace) rides along for retraction.
        let mut chars: Vec<char> = chars.to_vec();
        if !chars.contains(&'\u{8}') {
            chars.push('\u{8}');
        }

        let im = im_manager.as_ref().map(|m| m.get_input_method(&seat, &qh, ()));
        let mut keymap_fd = None;
        let vk = match vk_manager.as_ref() {
            Some(m) => {
                let vk = m.create_virtual_keyboard(&seat, &qh, ());
                let keymap = build_keymap(&chars);
                let fd = memfd_with(&keymap)?;
                // +1: compositors expect the NUL in the mapped size.
                vk.keymap(1, fd.as_fd(), keymap.len() as u32 + 1);
                vk.modifiers(0, 0, 0, 0);
                keymap_fd = Some(fd);
                Some(vk)
            }
            None => None,
        };

        let mut state = State {
            im_pending_active: false,
            im_active: false,
            im_serial: 0,
            im_unavailable: false,
            im_surrounding: false,
            im_activated_at: None,
        };
        // Settle registration; picks up immediate `unavailable` / `activate`.
        queue.roundtrip(&mut state).context("initial roundtrip")?;

        Ok(WaylandInjector {
            _conn: conn,
            queue,
            state,
            im,
            vk,
            chars,
            epoch: Instant::now(),
            _keymap_fd: keymap_fd,
        })
    }

    fn pump(&mut self) -> Result<()> {
        self.queue.roundtrip(&mut self.state).context("wayland roundtrip")?;
        Ok(())
    }

    fn im_usable(&self) -> bool {
        self.im.is_some() && self.state.im_active && !self.state.im_unavailable
    }

    /// Targets need a beat between activating us and their IME plumbing
    /// being ready; a commit sent in that window is silently dropped (seen
    /// with foot under sway). Irrelevant during interactive use — steering
    /// to the first letter takes far longer than this.
    fn settle_im(&mut self) -> Result<()> {
        const SETTLE: std::time::Duration = std::time::Duration::from_millis(60);
        if let Some(t) = self.state.im_activated_at {
            let age = t.elapsed();
            if age < SETTLE {
                std::thread::sleep(SETTLE - age);
                self.pump()?;
            }
        }
        Ok(())
    }

    fn now_ms(&self) -> u32 {
        self.epoch.elapsed().as_millis() as u32
    }

    fn tap(&mut self, c: char) -> Result<()> {
        let vk = self.vk.as_ref().context("no virtual keyboard")?;
        let code = self
            .chars
            .iter()
            .position(|&k| k == c)
            .with_context(|| format!("{c:?} is not in the injection keymap"))? as u32
            + 1;
        let t = self.now_ms();
        vk.key(t, code, KEY_PRESSED);
        vk.key(t, code, KEY_RELEASED);
        Ok(())
    }
}

impl Injector for WaylandInjector {
    fn commit(&mut self, text: &str) -> Result<()> {
        self.pump()?; // refresh im_active before choosing a path
        if self.im_usable() {
            self.settle_im()?;
            let im = self.im.as_ref().expect("im_usable checked");
            im.commit_string(text.to_string());
            im.commit(self.state.im_serial);
        } else {
            for c in text.chars() {
                self.tap(c)?;
            }
        }
        self.pump() // flush + surface protocol errors now, not next call
    }

    fn retract(&mut self, text: &str) -> Result<()> {
        self.pump()?;
        // delete_surrounding_text only where the target proved it maintains
        // surrounding text; terminals and other key-oriented clients ignore
        // the deletion, so give them backspaces even while the IM is active.
        if self.im_usable() && (self.state.im_surrounding || self.vk.is_none()) {
            self.settle_im()?;
            let im = self.im.as_ref().expect("im_usable checked");
            im.delete_surrounding_text(text.len() as u32, 0);
            im.commit(self.state.im_serial);
        } else {
            for _ in text.chars() {
                self.tap('\u{8}')?;
            }
        }
        self.pump()
    }

    fn capabilities(&self) -> Caps {
        Caps {
            name: "wayland",
            // Truthful only when an IM-activated field that reports
            // surrounding text is focused; everything else is backspace
            // synthesis.
            true_retraction: self.im_usable() && self.state.im_surrounding,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keymap_names_every_char() {
        let chars: Vec<char> = ('a'..='c').chain([' ', '\u{8}']).collect();
        let km = build_keymap(&chars);
        assert!(km.contains("<K0> = 9;"));
        assert!(km.contains("[ U0061 ]"));
        assert!(km.contains("[ U0020 ]"));
        assert!(km.contains("[ BackSpace ]"));
        assert!(km.contains("maximum = 14;"));
    }
}

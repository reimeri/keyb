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

const KEY_BACKSPACE: u32 = 14;
const KEY_TAB: u32 = 15;
const KEY_ENTER: u32 = 28;
const KEY_SPACE: u32 = 57;
const PINNED_KEYCODES: [u32; 4] = [KEY_BACKSPACE, KEY_TAB, KEY_ENTER, KEY_SPACE];

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
    /// Symbols in our uploaded keymap; `codes[i]` is the evdev keycode of
    /// `chars[i]`.
    chars: Vec<char>,
    codes: Vec<u32>,
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
        '\t' => "Tab".into(),
        '\u{8}' => "BackSpace".into(),
        _ => format!("U{:04X}", c as u32),
    }
}

/// Canonical evdev keycode for keys that Chromium resolves by physical
/// position instead of keysym (its editing/UI commands come from the
/// keycode; a BackSpace or space keysym on an arbitrary code does nothing
/// there — Chromium special-cases exactly enter, tab, space, backspace).
/// Firefox and terminals honor the keysym alone; pinning costs them nothing.
fn pinned_keycode(c: char) -> Option<u32> {
    match c {
        '\u{8}' => Some(KEY_BACKSPACE),
        '\t' => Some(KEY_TAB),
        '\n' => Some(KEY_ENTER),
        ' ' => Some(KEY_SPACE),
        _ => None,
    }
}

/// Evdev keycode per symbol: pinned keys sit on their canonical codes,
/// everything else fills sequentially from 1, skipping the pinned codes.
fn assign_keycodes(chars: &[char]) -> Vec<u32> {
    let mut next = 1u32;
    chars
        .iter()
        .map(|&c| match pinned_keycode(c) {
            Some(code) => code,
            None => {
                while PINNED_KEYCODES.contains(&next) {
                    next += 1;
                }
                let code = next;
                next += 1;
                code
            }
        })
        .collect()
}

/// Minimal xkb_v1 keymap: symbol `i` of `chars` lives on evdev keycode
/// `codes[i]` (xkb keycode `codes[i] + 8`).
fn build_keymap(chars: &[char], codes: &[u32]) -> String {
    let mut keycodes = String::new();
    let mut syms = String::new();
    for (i, (&c, &code)) in chars.iter().zip(codes).enumerate() {
        let xkb = code + 8;
        keycodes.push_str(&format!("        <K{i}> = {xkb};\n"));
        syms.push_str(&format!("        key <K{i}> {{ [ {} ] }};\n", keysym_name(c)));
    }
    format!(
        "xkb_keymap {{\n\
         xkb_keycodes \"keyb\" {{\n\
         \x20       minimum = 8;\n\
         \x20       maximum = {max};\n\
         {keycodes}\
         }};\n\
         xkb_types \"keyb\" {{ include \"complete\" }};\n\
         xkb_compatibility \"keyb\" {{ include \"complete\" }};\n\
         xkb_symbols \"keyb\" {{\n\
         {syms}\
         }};\n\
         }};\n",
        max = codes.iter().max().map_or(9, |m| m + 9),
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

        let codes = assign_keycodes(&chars);
        let im = im_manager.as_ref().map(|m| m.get_input_method(&seat, &qh, ()));
        let mut keymap_fd = None;
        let vk = match vk_manager.as_ref() {
            Some(m) => {
                let vk = m.create_virtual_keyboard(&seat, &qh, ());
                let keymap = build_keymap(&chars, &codes);
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
            codes,
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
            .map(|i| self.codes[i])
            .with_context(|| format!("{c:?} is not in the injection keymap"))?;
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
        // BackSpace taps whenever the virtual keyboard exists: browsers
        // (Chromium) silently drop delete_surrounding_text on many fields
        // even after reporting surrounding text, and its commit serial can
        // go stale mid-flight, while a key tap is never discarded. IM
        // deletion remains only for compositors without virtual-keyboard-v1.
        if self.vk.is_some() || !self.im_usable() {
            for _ in text.chars() {
                self.tap('\u{8}')?;
            }
        } else {
            self.settle_im()?;
            let im = self.im.as_ref().expect("im_usable checked");
            im.delete_surrounding_text(text.len() as u32, 0);
            im.commit(self.state.im_serial);
        }
        self.pump()
    }

    fn capabilities(&self) -> Caps {
        Caps {
            name: "wayland",
            // Truthful only on the no-vk fallback where retraction actually
            // goes through delete_surrounding_text; with a virtual keyboard
            // present, every retraction is backspace synthesis.
            true_retraction: self.vk.is_none()
                && self.im_usable()
                && self.state.im_surrounding,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keymap_names_every_char() {
        let chars: Vec<char> = ('a'..='c').chain([' ', '\u{8}']).collect();
        let km = build_keymap(&chars, &assign_keycodes(&chars));
        assert!(km.contains("<K0> = 9;"));
        assert!(km.contains("[ U0061 ]"));
        assert!(km.contains("[ U0020 ]"));
        assert!(km.contains("[ BackSpace ]"));
        assert!(km.contains("maximum = 66;"), "headroom above xkb 65 (space)");
    }

    #[test]
    fn special_keys_sit_on_canonical_keycodes() {
        // Chromium maps enter/tab/space/backspace by physical keycode, not
        // keysym; they must sit where a real keyboard puts them.
        let chars = vec!['a', '\n', 'b', '\u{8}', ' ', 'c'];
        let codes = assign_keycodes(&chars);
        assert_eq!(codes, vec![1, KEY_ENTER, 2, KEY_BACKSPACE, KEY_SPACE, 3]);
        let km = build_keymap(&chars, &codes);
        assert!(km.contains("<K1> = 36;")); // Return at xkb 28+8
        assert!(km.contains("key <K1> { [ Return ] };"));
        assert!(km.contains("<K3> = 22;")); // BackSpace at xkb 14+8
        assert!(km.contains("key <K3> { [ BackSpace ] };"));
        assert!(km.contains("<K4> = 65;")); // space at xkb 57+8
        assert!(km.contains("key <K4> { [ U0020 ] };"));
    }

    #[test]
    fn sequential_codes_skip_pinned_positions() {
        // Enough printables to walk past both reserved codes.
        let chars: Vec<char> = ('a'..='z').chain(['\u{8}', '\n']).collect();
        let codes = assign_keycodes(&chars);
        assert_eq!(codes.iter().filter(|&&c| c == KEY_BACKSPACE).count(), 1);
        assert_eq!(codes.iter().filter(|&&c| c == KEY_ENTER).count(), 1);
        let mut sorted = codes.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), codes.len(), "no keycode collisions");
    }
}

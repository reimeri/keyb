//! keyb-inject: system text output behind one trait (SPEC §8).
//!
//! Backends, auto-detected in order: Wayland (input-method-v2 preferred,
//! virtual-keyboard-v1 fallback, both over one connection), X11 XTEST, and an
//! in-process buffer as the last resort. The injector runs on its own worker
//! thread in keyb-app; nothing here touches the UI or the engine.

mod wayland;
mod x11;

use std::collections::VecDeque;

use anyhow::Result;

pub use wayland::WaylandInjector;
pub use x11::XtestInjector;

/// What the active backend can actually do (SPEC §8). keyb-app widens the
/// engine's retraction hysteresis when `true_retraction` is false.
#[derive(Clone, Copy, Debug)]
pub struct Caps {
    pub name: &'static str,
    /// Retraction edits the target's text properly (input-method
    /// delete_surrounding_text). False means synthesized backspaces, which
    /// mangle text if the target rewrites input (autocomplete, terminals in
    /// raw mode, ...).
    pub true_retraction: bool,
}

/// A system text-output backend. `retract` takes the exact text to remove
/// from the end of what was committed — backends need its char count
/// (backspace synthesis) or byte length (delete_surrounding_text).
pub trait Injector: Send {
    fn commit(&mut self, text: &str) -> Result<()>;
    fn retract(&mut self, text: &str) -> Result<()>;
    fn capabilities(&self) -> Caps;
}

/// Rolling record of the last `cap` commits (SPEC §8.3). Retraction beyond
/// what the ledger still tracks is refused — better to leave a character
/// standing in the target app than to backspace over text we never wrote.
pub struct RetractionLedger {
    chunks: VecDeque<String>,
    cap: usize,
}

impl Default for RetractionLedger {
    fn default() -> Self {
        RetractionLedger::new(32)
    }
}

impl RetractionLedger {
    pub fn new(cap: usize) -> Self {
        RetractionLedger {
            chunks: VecDeque::new(),
            cap,
        }
    }

    pub fn push(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.chunks.push_back(text.to_string());
        while self.chunks.len() > self.cap {
            self.chunks.pop_front();
        }
    }

    /// Whether `text` is the entire tracked suffix, without consuming it.
    /// Check before touching an injection backend; only take after it succeeds.
    pub fn can_take(&self, text: &str) -> bool {
        if text.is_empty() {
            return false;
        }
        let mut have = self.chunks.iter().rev().flat_map(|c| c.chars().rev());
        text.chars().rev().all(|want| have.next() == Some(want))
    }

    /// Try to take `text` off the tail. True means the ledger tracked it and
    /// the caller should retract it from the target; false means it fell off
    /// the rolling window (or never matched) and must be left alone.
    pub fn take(&mut self, text: &str) -> bool {
        if !self.can_take(text) {
            return false;
        }
        let mut need: Vec<char> = text.chars().collect();
        while let Some(mut last) = self.chunks.pop_back() {
            while !need.is_empty() {
                match last.pop() {
                    Some(c) => {
                        debug_assert_eq!(Some(&c), need.last());
                        need.pop();
                    }
                    None => break,
                }
            }
            if !last.is_empty() {
                self.chunks.push_back(last);
            }
            if need.is_empty() {
                return true;
            }
        }
        debug_assert!(need.is_empty(), "suffix was verified above");
        need.is_empty()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }
}

/// Last-resort backend: accumulates text internally and echoes to stderr.
/// Also the deterministic double for tests.
#[derive(Default)]
pub struct BufferInjector {
    pub text: String,
    quiet: bool,
}

impl BufferInjector {
    pub fn quiet() -> Self {
        BufferInjector {
            text: String::new(),
            quiet: true,
        }
    }
}

impl Injector for BufferInjector {
    fn commit(&mut self, text: &str) -> Result<()> {
        self.text.push_str(text);
        if !self.quiet {
            eprintln!("keyb[buffer]: {:?}", self.text);
        }
        Ok(())
    }

    fn retract(&mut self, text: &str) -> Result<()> {
        for _ in text.chars() {
            self.text.pop();
        }
        if !self.quiet {
            eprintln!("keyb[buffer]: {:?}", self.text);
        }
        Ok(())
    }

    fn capabilities(&self) -> Caps {
        Caps {
            name: "buffer",
            true_retraction: true,
        }
    }
}

/// Pick a backend (SPEC §8): Wayland → X11 → buffer. `chars` is the set the
/// virtual-keyboard keymap must cover (the alphabet). `KEYB_INJECT` forces
/// one of `wayland`, `wayland-vk` (skip input-method), `x11`, `buffer`.
pub fn detect(chars: &[char]) -> Box<dyn Injector> {
    let forced = std::env::var("KEYB_INJECT").ok();
    let try_wayland = |chars: &[char]| {
        let r = if forced.as_deref() == Some("wayland-vk") {
            WaylandInjector::vk_only(chars)
        } else {
            WaylandInjector::new(chars)
        };
        match r {
            Ok(i) => Some(Box::new(i) as Box<dyn Injector>),
            Err(e) => {
                eprintln!("keyb: wayland injection unavailable: {e}");
                None
            }
        }
    };
    let try_x11 = || match XtestInjector::new() {
        Ok(i) => Some(Box::new(i) as Box<dyn Injector>),
        Err(e) => {
            eprintln!("keyb: x11 injection unavailable: {e}");
            None
        }
    };
    let picked = match forced.as_deref() {
        Some("wayland") | Some("wayland-vk") => try_wayland(chars),
        Some("x11") => try_x11(),
        Some("buffer") | Some("none") => None,
        Some(other) => {
            eprintln!("keyb: unknown KEYB_INJECT={other}, using buffer");
            None
        }
        None => {
            let wl = if std::env::var_os("WAYLAND_DISPLAY").is_some() {
                try_wayland(chars)
            } else {
                None
            };
            wl.or_else(|| {
                if std::env::var_os("DISPLAY").is_some() {
                    try_x11()
                } else {
                    None
                }
            })
        }
    };
    picked.unwrap_or_else(|| Box::new(BufferInjector::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_pops_across_chunk_boundaries() {
        let mut l = RetractionLedger::new(32);
        l.push("th");
        l.push("e");
        assert!(l.take("e"));
        assert!(l.take("h"));
        assert!(l.take("t"));
        assert!(l.is_empty());
        assert!(!l.take("x"), "empty ledger must refuse");
    }

    #[test]
    fn ledger_refuses_mismatch_and_overflow() {
        let mut l = RetractionLedger::new(2);
        l.push("a");
        l.push("b");
        l.push("c"); // "a" falls off the window
        assert!(!l.take("x"), "mismatched suffix refused");
        assert!(l.take("bc"), "tracked suffix accepted");
        assert!(!l.take("a"), "beyond the window refused");
    }

    #[test]
    fn ledger_suffix_probe_does_not_consume_text() {
        let mut l = RetractionLedger::new(32);
        l.push("th");
        l.push("é");
        assert!(!l.can_take("xé"));
        assert!(!l.can_take(""));
        assert!(l.can_take("hé"));
        assert!(l.can_take("hé"));
        assert!(l.take("hé"));
        assert!(l.take("t"));
        assert!(l.is_empty());
    }

    #[test]
    fn buffer_injector_roundtrip() {
        let mut b = BufferInjector::quiet();
        b.commit("hi ").unwrap();
        b.commit("there").unwrap();
        b.retract("ere").unwrap();
        assert_eq!(b.text, "hi th");
    }
}

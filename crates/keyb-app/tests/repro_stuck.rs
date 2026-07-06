//! Reproduction of the "no boxes / stuck" report while typing "The man".
//!
//! A real LLM gives sentence-initial capitals real mass, so we wrap PPM with a
//! model that boosts the exact target character at each step enough to steer
//! into it — including the capital via its group box — then check the phrase
//! completes without the tree wedging.

use std::cell::RefCell;

use keyb_core::sim::{run_oracle, OracleParams};
use keyb_core::{Alphabet, Engine, EngineParams, LanguageModel};
use keyb_lm_ppm::PpmModel;

/// Wraps PPM but pours a big fixed share onto whatever the *next target*
/// character is for the current committed length, so even a grouped capital is
/// reachable. This mimics a strong model without needing the GGUF.
struct Guided {
    ppm: PpmModel,
    alphabet: Alphabet,
    target: Vec<u8>,
    committed_len: RefCell<usize>,
}

impl Guided {
    fn new(alphabet: &Alphabet, target: &str) -> Self {
        Guided {
            ppm: PpmModel::seeded(alphabet),
            target: alphabet.encode(target),
            alphabet: alphabet.clone(),
            committed_len: RefCell::new(0),
        }
    }
}

impl LanguageModel for Guided {
    fn predict(&mut self, context: &[u8]) -> Vec<f64> {
        let mut p = self.ppm.predict(context);
        // The target char for a context this length (context is absolute here
        // because the engine passes document-start contexts to predict).
        if context.len() < self.target.len() && context == &self.target[..context.len()] {
            let want = self.target[context.len()] as usize;
            for v in p.iter_mut() {
                *v *= 0.1;
            }
            p[want] += 0.9;
            let s: f64 = p.iter().sum();
            for v in p.iter_mut() {
                *v /= s;
            }
        }
        p
    }
    fn learn(&mut self, c: &[u8], s: u8) {
        self.ppm.learn(c, s);
        *self.committed_len.borrow_mut() += 1;
    }
    fn unlearn(&mut self, c: &[u8], s: u8) {
        self.ppm.unlearn(c, s);
    }
    fn alphabet_len(&self) -> usize {
        self.alphabet.len()
    }
}

fn try_phrase(text: &str, rebase: f64) -> (bool, String) {
    let alphabet = Alphabet::english_full();
    let lm = Guided::new(&alphabet, text);
    let mut params = EngineParams::default();
    params.rebase_view_w = rebase;
    let mut e = Engine::new(alphabet, Box::new(lm), params);
    e.set_viewport_px(900.0);
    e.speed_bits = 6.0;
    let r = run_oracle(&mut e, text, &OracleParams { max_secs: 400.0, ..Default::default() }, 5);
    (r.completed, e.committed_text())
}

#[test]
fn guided_the_man_completes() {
    let (ok, got) = try_phrase("The man", 1e-6);
    eprintln!("The man -> {got:?} ok={ok}");
    assert!(ok, "stuck typing 'The man': got {got:?}");
}

#[test]
fn guided_the_man_completes_forced_rebase() {
    let (ok, got) = try_phrase("The man", 0.5);
    eprintln!("The man (forced rebase) -> {got:?} ok={ok}");
    assert!(ok, "stuck typing 'The man' under forced rebase: got {got:?}");
}

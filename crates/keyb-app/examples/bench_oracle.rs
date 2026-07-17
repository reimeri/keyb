//! Oracle throughput benchmark (SPEC §11): WPM on held-out English per model.
//!
//! The M1 oracle test sentence closely resembles the PPM seed corpus; these
//! sentences are held out so PPM vs LLM is a fair comparison and the M2 exit
//! criterion ("oracle ≥ 25 WPM on held-out English") is measured honestly.
//!
//! Usage: bench_oracle [ppm|llama|both]
//!
//! The llama run paces the simulation to the wall clock (inference races real
//! frames, exactly as it would for a human), so it takes as long as the
//! simulated typing. It also reports the share of node expansions that were
//! served with PPM-fallback sizing (§10 budget: < 5 % during steady steering).

use keyb_core::sim::{run_oracle, OracleParams};
use keyb_core::{Alphabet, Engine, EngineParams, LanguageModel};
use keyb_lm_llama::{LlamaConfig, LlamaLm};
use keyb_lm_ppm::PpmModel;

pub const HELD_OUT: &[&str] = &[
    "please call stella and ask her to bring these things with her from the store",
    "the council will meet on thursday to discuss the new housing proposal",
    "she poured the coffee slowly while reading the letter from her brother",
    "our flight leaves early in the morning so we should pack tonight",
    "the museum was quiet except for a group of students sketching the statues",
];

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Wraps an LM and counts predict() calls vs. later upgrades: an upgrade
/// means that expansion was first served with PPM-fallback sizing (§10).
struct Instrumented<L> {
    inner: L,
    requests: Arc<AtomicUsize>,
    upgrades: Arc<AtomicUsize>,
}

impl<L: LanguageModel> LanguageModel for Instrumented<L> {
    fn predict(&mut self, context: &[u8]) -> Vec<f64> {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.inner.predict(context)
    }

    fn poll_upgrades(&mut self) -> Vec<(Vec<u8>, Vec<f64>)> {
        let u = self.inner.poll_upgrades();
        self.upgrades.fetch_add(u.len(), Ordering::Relaxed);
        u
    }

    fn frame_hint(&mut self, ctx: &[u8]) {
        self.inner.frame_hint(ctx);
    }

    fn release(&mut self, ctx: &[u8]) {
        self.inner.release(ctx);
    }

    fn learn(&mut self, context: &[u8], sym: u8) {
        self.inner.learn(context, sym);
    }

    fn unlearn(&mut self, context: &[u8], sym: u8) {
        self.inner.unlearn(context, sym);
    }

    fn alphabet_len(&self) -> usize {
        self.inner.alphabet_len()
    }
}

fn run(name: &str, mk: impl Fn(&Alphabet) -> Box<dyn LanguageModel>, realtime: bool) {
    println!("== {name} ==");
    let mut wpms = Vec::new();
    let mut all_completed = true;
    for (i, target) in HELD_OUT.iter().enumerate() {
        let alphabet = Alphabet::english_v1();
        let (requests, upgrades) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let lm = Box::new(Instrumented {
            inner: mk(&alphabet),
            requests: requests.clone(),
            upgrades: upgrades.clone(),
        });
        let mut e = Engine::new(alphabet, lm, EngineParams::default());
        e.set_viewport_px(800.0);
        e.speed_bits = 6.0;
        let params = OracleParams {
            realtime,
            ..Default::default()
        };
        let r = run_oracle(&mut e, target, &params, 42 + i as u64);
        let (req, upg) = (
            requests.load(Ordering::Relaxed),
            upgrades.load(Ordering::Relaxed),
        );
        println!(
            "[{}] completed={} chars={:3} secs={:6.1} wpm={:5.1} fallback={:4.1}%  \"{}...\"",
            i,
            r.completed,
            r.chars,
            r.secs,
            r.wpm,
            100.0 * upg as f64 / req.max(1) as f64,
            &target[..target.len().min(40)]
        );
        all_completed &= r.completed;
        wpms.push(r.wpm);
    }
    let mean = wpms.iter().sum::<f64>() / wpms.len() as f64;
    println!("{name} held-out mean: {mean:.1} WPM  (all completed: {all_completed})\n");
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "both".into());
    if mode == "ppm" || mode == "both" {
        run("ppm", |a| Box::new(PpmModel::seeded(a)), false);
    }
    if mode == "llama" || mode == "both" {
        let cfg = LlamaConfig::default();
        if !cfg.model_path.exists() {
            eprintln!("no model at {}", cfg.model_path.display());
            std::process::exit(1);
        }
        run(
            "llama (realtime)",
            move |a| Box::new(LlamaLm::new(a, cfg.clone()).expect("model loads")),
            true,
        );
    }
}

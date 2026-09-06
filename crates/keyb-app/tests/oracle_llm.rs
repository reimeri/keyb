//! M2 exit criterion (SPEC §12): oracle ≥ 25 WPM on held-out English with
//! the LLM backend. Paced to the wall clock so inference latency is part of
//! the measurement. Skipped when the model file is absent.

use keyb_core::sim::{run_oracle, OracleParams};
use keyb_core::{Alphabet, Engine, EngineParams, LanguageModel};
use keyb_lm_llama::{LlamaConfig, LlamaLm};

#[test]
fn oracle_reaches_m2_wpm_target_on_held_out_text() {
    let cfg = LlamaConfig::default();
    if !cfg.model_path.exists() {
        eprintln!("skipping: no model at {}", cfg.model_path.display());
        return;
    }
    let alphabet = Alphabet::english_v1();
    let lm = LlamaLm::new(&alphabet, cfg).expect("model must load");
    let mut e = Engine::new(alphabet, Box::new(lm), EngineParams::default());
    e.set_viewport_px(800.0);
    e.speed_bits = 6.0;
    // Held out: not in the PPM seed corpus, not an M1 test sentence.
    let target = "the council will meet on thursday to discuss the new housing proposal";
    let params = OracleParams {
        realtime: true,
        ..Default::default()
    };
    let r = run_oracle(&mut e, target, &params, 43);
    eprintln!(
        "llm oracle: completed={} chars={} secs={:.1} wpm={:.1}",
        r.completed, r.chars, r.secs, r.wpm
    );
    assert!(r.completed, "oracle did not finish: {r:?}");
    assert!(
        r.wpm >= 25.0,
        "M2 target is 25 WPM held-out, got {:.1}",
        r.wpm
    );
}

/// Wait for the worker's real answer for `ctx` (the first `predict` is a
/// fallback by design, SPEC §6.3), i.e. the upgrade that follows it.
fn wait_for_upgrade(lm: &mut LlamaLm, ctx: &[u8]) -> Vec<f64> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        for (c, d) in lm.poll_upgrades() {
            if c == ctx {
                return d;
            }
        }
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for the worker's answer"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// SPEC §6.8: a captured context block must actually reach the model.
/// That context *helps* typing speed is measured manually with bench_oracle,
/// not asserted on a 0.5B model — this asserts the plumbing.
#[test]
fn context_block_reaches_the_model() {
    let cfg = LlamaConfig::default();
    if !cfg.model_path.exists() {
        eprintln!("skipping: no model at {}", cfg.model_path.display());
        return;
    }
    let alphabet = Alphabet::english_v1();
    let mut lm = LlamaLm::new(&alphabet, cfg).expect("model must load");

    // Same question with and without the block: the distribution must move.
    let _ = lm.predict(b"");
    let without = wait_for_upgrade(&mut lm, b"");
    lm.set_context(Some(
        "alice: are you coming to the meeting on thursday?\n\
         bob: yes, the council will discuss the housing proposal there\n\
         alice:",
    ));
    let _ = lm.predict(b"");
    let with = wait_for_upgrade(&mut lm, b"");
    let tv: f64 = 0.5
        * without
            .iter()
            .zip(&with)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();
    eprintln!("context total-variation distance: {tv:.3}");
    assert!(
        tv > 0.01,
        "context block must change predictions measurably (tv={tv:.4})"
    );
}

//! M2 exit criterion (SPEC §12): oracle ≥ 25 WPM on held-out English with
//! the LLM backend. Paced to the wall clock so inference latency is part of
//! the measurement. Skipped when the model file is absent.

use keyb_core::sim::{run_oracle, OracleParams};
use keyb_core::{Alphabet, Engine, EngineParams};
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

//! M1 exit criterion (SPEC §12): the oracle steerer must reach ≥ 15 WPM with
//! the PPM model, motor noise, and reaction delay.

use keyb_core::sim::{run_oracle, OracleParams};
use keyb_core::{Alphabet, Engine, EngineParams};
use keyb_lm_ppm::PpmModel;

fn engine() -> Engine {
    let alphabet = Alphabet::english_v1();
    let lm = PpmModel::seeded(&alphabet);
    Engine::new(alphabet, Box::new(lm), EngineParams::default())
}

#[test]
fn oracle_reaches_m1_wpm_target() {
    let mut e = engine();
    e.set_viewport_px(800.0);
    e.speed_bits = 6.0;
    let target = "the weather today is quite nice and i would like to go for a walk in the park";
    let r = run_oracle(&mut e, target, &OracleParams::default(), 42);
    eprintln!(
        "oracle: completed={} chars={} secs={:.1} wpm={:.1}",
        r.completed, r.chars, r.secs, r.wpm
    );
    assert!(r.completed, "oracle did not finish: {r:?}");
    assert!(r.wpm >= 15.0, "M1 target is 15 WPM, got {:.1}", r.wpm);
}

#[test]
fn grouped_alphabet_keeps_wpm_on_lowercase() {
    // The full (grouped) alphabet diverts only a sliver of mass into the
    // capitals/digits/symbols group boxes, so lowercase throughput must stay
    // at the M1 target — grouping is structure, not a speed tax (SPEC §5.4).
    let alphabet = Alphabet::english_full();
    let lm = PpmModel::seeded(&alphabet);
    let mut e = Engine::new(alphabet, Box::new(lm), EngineParams::default());
    e.set_viewport_px(800.0);
    e.speed_bits = 6.0;
    let target = "the weather today is quite nice and i would like to go for a walk in the park";
    let r = run_oracle(&mut e, target, &OracleParams::default(), 42);
    eprintln!(
        "grouped oracle: wpm={:.1} chars={} secs={:.1}",
        r.wpm, r.chars, r.secs
    );
    assert!(
        r.completed,
        "oracle did not finish on grouped alphabet: {r:?}"
    );
    assert!(
        r.wpm >= 15.0,
        "grouped alphabet must hold ≥ 15 WPM, got {:.1}",
        r.wpm
    );
}

#[test]
fn oracle_handles_unseeded_text() {
    // Text with rare letters — checks the floor keeps everything reachable.
    let mut e = engine();
    e.set_viewport_px(800.0);
    e.speed_bits = 6.0;
    let r = run_oracle(
        &mut e,
        "quartz jukebox vex",
        &OracleParams {
            max_secs: 300.0,
            ..Default::default()
        },
        7,
    );
    eprintln!("rare-letter oracle: {r:?}");
    assert!(r.completed, "oracle did not finish rare-letter text: {r:?}");
}

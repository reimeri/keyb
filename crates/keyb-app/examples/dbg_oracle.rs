//! Oracle sweep: WPM vs speed cap. Tuning aid for dynamics/prediction changes.

use keyb_core::sim::{run_oracle, OracleParams};
use keyb_core::{Alphabet, Engine, EngineParams};
use keyb_lm_ppm::PpmModel;

fn main() {
    let target = "the weather today is quite nice and i would like to go for a walk in the park";
    for speed in [3.0, 4.5, 6.0, 8.0, 10.0] {
        let alphabet = Alphabet::english_v1();
        let lm = PpmModel::seeded(&alphabet);
        let mut e = Engine::new(alphabet, Box::new(lm), EngineParams::default());
        e.set_viewport_px(800.0);
        e.speed_bits = speed;
        let r = run_oracle(&mut e, target, &OracleParams::default(), 42);
        println!(
            "speed {speed:4.1} bits/s -> completed={} chars={} secs={:6.1} wpm={:5.1}",
            r.completed, r.chars, r.secs, r.wpm
        );
    }
}

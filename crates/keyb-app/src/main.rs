use anyhow::Result;
use keyb_core::{Alphabet, Engine, EngineParams};
use keyb_lm_ppm::PpmModel;

fn main() -> Result<()> {
    let alphabet = Alphabet::english_v1();
    let lm = PpmModel::seeded(&alphabet);
    let engine = Engine::new(alphabet, Box::new(lm), EngineParams::default());
    keyb_ui::run(engine)
}

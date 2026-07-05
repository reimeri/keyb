use anyhow::Result;
use keyb_core::{Alphabet, Engine, EngineParams, LanguageModel};
use keyb_lm_llama::{LlamaConfig, LlamaLm};
use keyb_lm_ppm::PpmModel;

/// LLM if the model file is present (`KEYB_MODEL` overrides the default
/// path), PPM alone otherwise — the interface works either way, the boxes
/// are just less prescient.
fn language_model(alphabet: &Alphabet) -> Box<dyn LanguageModel> {
    let mut cfg = LlamaConfig::default();
    if let Some(p) = std::env::var_os("KEYB_MODEL") {
        cfg.model_path = p.into();
    }
    if !cfg.model_path.exists() {
        eprintln!(
            "keyb: no model at {} — running on PPM only",
            cfg.model_path.display()
        );
        return Box::new(PpmModel::seeded(alphabet));
    }
    match LlamaLm::new(alphabet, cfg) {
        Ok(lm) => Box::new(lm),
        Err(e) => {
            eprintln!("keyb: {e} — running on PPM only");
            Box::new(PpmModel::seeded(alphabet))
        }
    }
}

fn main() -> Result<()> {
    let alphabet = Alphabet::english_v1();
    let lm = language_model(&alphabet);
    let engine = Engine::new(alphabet, lm, EngineParams::default());
    keyb_ui::run(engine)
}

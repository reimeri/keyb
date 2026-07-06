use anyhow::Result;
use keyb_core::{Alphabet, Engine, EngineParams, LanguageModel, TextEvent};
use keyb_inject::{Injector, RetractionLedger};
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

/// Injection worker (SPEC §8): owns the backend and the rolling retraction
/// ledger on its own thread, so a slow compositor roundtrip can never stall
/// the frame loop. Returns the sender the UI feeds and the backend's
/// capabilities (read once at startup for the hysteresis decision).
fn injection_worker(
    mut injector: Box<dyn Injector>,
) -> (crossbeam_channel::Sender<TextEvent>, keyb_inject::Caps) {
    let caps = injector.capabilities();
    let (tx, rx) = crossbeam_channel::unbounded::<TextEvent>();
    std::thread::Builder::new()
        .name("keyb-inject".into())
        .spawn(move || {
            let mut ledger = RetractionLedger::default();
            for ev in rx {
                let r = match &ev {
                    TextEvent::Commit(text) => {
                        ledger.push(text);
                        injector.commit(text)
                    }
                    // Only retract text the ledger still tracks (SPEC §8.3);
                    // anything older stays put in the target app.
                    TextEvent::Retract(text) => {
                        if ledger.take(text) {
                            injector.retract(text)
                        } else {
                            Ok(())
                        }
                    }
                };
                if let Err(e) = r {
                    eprintln!("keyb: injection failed: {e:#}");
                }
            }
        })
        .expect("spawn injection worker");
    (tx, caps)
}

fn main() -> Result<()> {
    let alphabet = Alphabet::english_v1();
    let lm = language_model(&alphabet);

    let chars: Vec<char> = (0..alphabet.len() as u8).map(|s| alphabet.char_of(s)).collect();
    let (tx, caps) = injection_worker(keyb_inject::detect(&chars));
    eprintln!(
        "keyb: injecting via {} ({} retraction)",
        caps.name,
        if caps.true_retraction { "true" } else { "backspace" }
    );

    let mut params = EngineParams::default();
    if !caps.true_retraction {
        // Backspace synthesis into an unknown app: demand a firmer back-zoom
        // before retracting (SPEC §8.3).
        params.retract_eps = 0.10;
    }

    let engine = Engine::new(alphabet, lm, params);
    keyb_ui::run(engine, Box::new(move |ev| {
        let _ = tx.send(ev);
    }))
}

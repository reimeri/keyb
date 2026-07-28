use anyhow::Result;
use keyb_core::{Alphabet, Engine, EngineParams, LanguageModel, TextEvent};
use keyb_inject::{Injector, RetractionLedger};
use keyb_lm_llama::{LlamaConfig, LlamaLm};
use keyb_lm_ppm::{Personalized, PpmModel};
use std::ffi::OsString;
use std::path::PathBuf;

mod model_download;

fn configured_model(model_override: Option<OsString>) -> (LlamaConfig, bool) {
    let mut cfg = LlamaConfig::default();
    let managed_default = model_override.is_none();
    if let Some(path) = model_override {
        cfg.model_path = path.into();
    }
    (cfg, managed_default)
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

/// `$XDG_DATA_HOME/keyb/user_model.v1.bin` — the persistent user-vocabulary
/// store, alongside the downloaded models (SPEC §9). Same base-dir resolution
/// as `keyb_lm_llama::default_model_path`.
fn user_model_path() -> PathBuf {
    let data = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_default();
    data.join("keyb/user_model.v1.bin")
}

/// Provision and load the default LLM (`KEYB_MODEL` selects a user-managed
/// file), falling back to PPM so download and load failures never prevent use.
fn language_model(alphabet: &Alphabet) -> Box<dyn LanguageModel> {
    let (cfg, managed_default) = configured_model(std::env::var_os("KEYB_MODEL"));
    if managed_default && !env_flag("KEYB_NO_MODEL_DOWNLOAD") {
        if let Err(error) = model_download::ensure_default_model(&cfg.model_path) {
            eprintln!("keyb: model download failed: {error:#} — running on PPM only");
            return Box::new(PpmModel::seeded(alphabet));
        }
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
) -> (
    crossbeam_channel::Sender<TextEvent>,
    keyb_inject::Caps,
    std::thread::JoinHandle<()>,
) {
    let caps = injector.capabilities();
    let (tx, rx) = crossbeam_channel::unbounded::<TextEvent>();
    let worker = std::thread::Builder::new()
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
    (tx, caps, worker)
}

fn main() -> Result<()> {
    // The full alphabet groups capitals, digits, and symbols under expandable
    // boxes (SPEC §5.4); its first 30 leaves match english_v1 exactly.
    let alphabet = Alphabet::english_full();
    // Wrap the base model in the persistent, decaying user-vocabulary layer so
    // words the user actually types get easier to enter over time (and fade
    // once unused). Opt out with KEYB_NO_PERSONALIZE.
    let lm: Box<dyn LanguageModel> = if env_flag("KEYB_NO_PERSONALIZE") {
        language_model(&alphabet)
    } else {
        Box::new(Personalized::load_or_new(
            alphabet.len(),
            language_model(&alphabet),
            user_model_path(),
        ))
    };

    let chars: Vec<char> = (0..alphabet.len() as u8)
        .map(|s| alphabet.char_of(s))
        .collect();
    let (tx, caps, injection_thread) = injection_worker(keyb_inject::detect(&chars));
    eprintln!(
        "keyb: injecting via {} ({} retraction)",
        caps.name,
        if caps.true_retraction {
            "true"
        } else {
            "backspace"
        }
    );

    let mut params = EngineParams::default();
    if !caps.true_retraction {
        // Backspace synthesis into an unknown app: demand a firmer back-zoom
        // before retracting (SPEC §8.3).
        params.retract_eps = 0.10;
    }
    // Dasher-style auto speed control (SPEC §5.2), opt-in: the cap ramps up
    // while steering is smooth and is cut on jitter/reversals. Off by default
    // so the established fixed-speed feel is unchanged.
    if env_flag("KEYB_ADAPTIVE") {
        params.adaptive_speed = true;
    }

    let engine = Engine::new(alphabet, lm, params);
    let ui_result = keyb_ui::run(
        engine,
        Box::new(move |ev| {
            let _ = tx.send(ev);
        }),
    );
    // UI teardown drops the final sender. Drain any text already queued before
    // allowing process exit so Close is a clean shutdown of the injector too.
    injection_thread
        .join()
        .map_err(|_| anyhow::anyhow!("injection worker panicked during shutdown"))?;
    ui_result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_model_paths_are_not_managed() {
        let (cfg, managed) = configured_model(Some(OsString::from("/tmp/custom.gguf")));
        assert_eq!(cfg.model_path, std::path::Path::new("/tmp/custom.gguf"));
        assert!(!managed);
    }

    #[test]
    fn default_model_path_is_managed() {
        let (cfg, managed) = configured_model(None);
        assert_eq!(cfg.model_path, keyb_lm_llama::default_model_path());
        assert!(managed);
    }
}

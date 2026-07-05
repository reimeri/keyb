//! Integration tests against the real GGUF model + tokenizer (SPEC §11).
//! Skipped (pass trivially) when the default model file is absent, so CI
//! without the ~500 MB download still runs the pure tests.

use std::time::{Duration, Instant};

use keyb_core::{Alphabet, LanguageModel};
use keyb_lm_llama::{LlamaConfig, LlamaLm};

fn lm_or_skip() -> Option<(LlamaLm, Alphabet)> {
    let cfg = LlamaConfig::default();
    if !cfg.model_path.exists() {
        eprintln!("skipping: no model at {}", cfg.model_path.display());
        return None;
    }
    let alphabet = Alphabet::english_v1();
    let lm = LlamaLm::new(&alphabet, cfg).expect("model must load");
    Some((lm, alphabet))
}

/// Request a context, wait for the async upgrade, return it.
fn upgraded(lm: &mut LlamaLm, ctx: &[u8], deadline: Duration) -> Vec<f64> {
    let fallback = lm.predict(ctx);
    assert_eq!(fallback.len(), lm.alphabet_len());
    let t0 = Instant::now();
    while t0.elapsed() < deadline {
        for (uctx, dist) in lm.poll_upgrades() {
            if uctx == ctx {
                return dist;
            }
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    panic!("no upgrade for {ctx:?} within {deadline:?}");
}

#[test]
fn upgrades_arrive_and_are_valid_distributions() {
    let Some((mut lm, alphabet)) = lm_or_skip() else { return };
    for text in ["", "the weather is ", "hello wor"] {
        let ctx = alphabet.encode(text);
        let dist = upgraded(&mut lm, &ctx, Duration::from_secs(10));
        assert_eq!(dist.len(), alphabet.len());
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "{text:?}: sum={sum}");
        assert!(
            dist.iter().all(|&p| p > 0.0),
            "{text:?}: floor must keep every symbol reachable"
        );
    }
}

#[test]
fn llm_beats_uniform_on_obvious_continuations() {
    let Some((mut lm, alphabet)) = lm_or_skip() else { return };
    // Mid-token healing: "nic" inside "...is nice" should strongly favor 'e'.
    let ctx = alphabet.encode("the weather today is really nic");
    let dist = upgraded(&mut lm, &ctx, Duration::from_secs(10));
    let e = dist[alphabet.index_of('e').unwrap() as usize];
    assert!(e > 0.5, "p('e' | '...really nic') = {e}, expected dominant");
    // Word boundary: after "thank", 's' or space should dwarf 'q'.
    let ctx = alphabet.encode("thank");
    let dist = upgraded(&mut lm, &ctx, Duration::from_secs(10));
    let s = dist[alphabet.index_of('s').unwrap() as usize];
    let q = dist[alphabet.index_of('q').unwrap() as usize];
    assert!(s > 20.0 * q, "s={s} q={q}");
}

#[test]
fn request_to_upgrade_latency_within_budget() {
    let Some((mut lm, alphabet)) = lm_or_skip() else { return };
    // Warm up (first request pays branch construction for the whole prefix).
    let _ = upgraded(&mut lm, &alphabet.encode("we went to"), Duration::from_secs(10));
    // Steady state: extend one symbol at a time, like steering does.
    let mut lats = Vec::new();
    let base = alphabet.encode("we went to the market and bought some");
    for k in 0..8 {
        let ctx = &base[..base.len() - 8 + k];
        let t0 = Instant::now();
        let _ = upgraded(&mut lm, ctx, Duration::from_secs(10));
        lats.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    lats.sort_by(f64::total_cmp);
    let p95 = lats[((lats.len() as f64 * 0.95) as usize).min(lats.len() - 1)];
    eprintln!("request→upgrade latencies (ms): {lats:.1?}, p95={p95:.1}");
    // SPEC §10: ≤ 80 ms p95 on the target machine; allow CI slack ×2.
    assert!(p95 < 160.0, "p95 latency {p95:.1} ms exceeds budget");
}

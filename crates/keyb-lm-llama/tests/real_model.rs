//! Integration tests against the real GGUF model + tokenizer (SPEC §11).
//! Skipped (pass trivially) when the default model file is absent, so CI
//! without the ~500 MB download still runs the pure tests.

use std::time::{Duration, Instant};

use keyb_core::{Alphabet, LanguageModel};
use keyb_lm_llama::{LlamaConfig, LlamaLm};

fn lm_or_skip() -> Option<(LlamaLm, Alphabet)> {
    lm_with_config_or_skip(LlamaConfig::default())
}

fn lm_with_config_or_skip(cfg: LlamaConfig) -> Option<(LlamaLm, Alphabet)> {
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
    let Some((mut lm, alphabet)) = lm_or_skip() else {
        return;
    };
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
    let Some((mut lm, alphabet)) = lm_or_skip() else {
        return;
    };
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
fn healing_preserves_tokens_that_span_canonical_boundaries() {
    let Some((mut lm, alphabet)) = lm_or_skip() else {
        return;
    };
    for text in ["hey, what are you doing? i'm tryin", "i'm goin"] {
        let ctx = alphabet.encode(text);
        let dist = upgraded(&mut lm, &ctx, Duration::from_secs(10));
        let g = dist[alphabet.index_of('g').unwrap() as usize];
        let space = dist[alphabet.index_of(' ').unwrap() as usize];
        let apostrophe = dist[alphabet.index_of('\'').unwrap() as usize];
        eprintln!("{text:?}: g={g:.4} space={space:.4} apostrophe={apostrophe:.4}");
        assert!(
            g > 0.8 && g > 20.0 * space && g > 20.0 * apostrophe,
            "{text:?}: g={g} space={space} apostrophe={apostrophe}"
        );
    }
}

#[test]
fn request_to_upgrade_latency_within_budget() {
    let Some((mut lm, alphabet)) = lm_or_skip() else {
        return;
    };
    // Warm up (first request pays branch construction for the whole prefix).
    let _ = upgraded(
        &mut lm,
        &alphabet.encode("we went to"),
        Duration::from_secs(10),
    );
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

/// SPEC §6.8: a captured block must move the empty-context distribution, a
/// tail-trimmed over-budget block must still upgrade, and clearing restores
/// preamble-only conditioning.
#[test]
fn context_block_reaches_the_model_and_clears() {
    let Some((mut lm, alphabet)) = lm_or_skip() else {
        return;
    };
    let without = upgraded(&mut lm, b"", Duration::from_secs(10));
    lm.set_context(Some(
        "alice: are you coming to the meeting on thursday?\n\
         bob: yes, the council will discuss the housing proposal there\n\
         alice:",
    ));
    let with = upgraded(&mut lm, b"", Duration::from_secs(15));
    let tv = 0.5
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

    // Over the worker's token budget: tail-trim rather than fail. Prefix
    // rebuild of ~1k tokens is slower than a short block.
    let long = "the council will meet on thursday.\n".repeat(200);
    lm.set_context(Some(&long));
    let dist = upgraded(&mut lm, b"", Duration::from_secs(60));
    assert_eq!(dist.len(), alphabet.len());
    let sum: f64 = dist.iter().sum();
    assert!((sum - 1.0).abs() < 1e-9, "over-budget block: sum={sum}");
    assert!(dist.iter().all(|&p| p > 0.0));

    lm.set_context(None);
    let cleared = upgraded(&mut lm, b"", Duration::from_secs(15));
    let tv_clear = 0.5
        * without
            .iter()
            .zip(&cleared)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();
    eprintln!("cleared vs original total-variation: {tv_clear:.4}");
    assert!(
        tv_clear < 0.01,
        "clearing must restore preamble-only conditioning (tv={tv_clear:.4})"
    );
}

fn total_variation(left: &[f64], right: &[f64]) -> f64 {
    0.5 * left
        .iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
}

#[test]
fn full_custom_prompt_reaches_model_and_survives_context_changes() {
    let Some((mut default_lm, _)) = lm_or_skip() else {
        return;
    };
    let default = upgraded(&mut default_lm, b"", Duration::from_secs(15));
    drop(default_lm);
    let cfg = LlamaConfig {
        system_prompt: "A shopping list of fruit:\n- apples\n- bananas\n- oranges\n-".into(),
        ..LlamaConfig::default()
    };
    let (mut lm, _) = lm_with_config_or_skip(cfg).unwrap();
    let custom = upgraded(&mut lm, b"", Duration::from_secs(15));
    assert!(
        total_variation(&default, &custom) > 0.01,
        "custom prompt must affect predictions"
    );

    for context in [
        "alice: what should we buy for dinner?".to_owned(),
        "the council meets on thursday.\n".repeat(200),
    ] {
        lm.set_context(Some(&context));
        let with_context = upgraded(&mut lm, b"", Duration::from_secs(60));
        assert!((with_context.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        lm.set_context(None);
        let cleared = upgraded(&mut lm, b"", Duration::from_secs(15));
        assert!(
            total_variation(&custom, &cleared) < 0.01,
            "clearing context must restore the custom prompt, not the built-in one"
        );
    }
}

#[test]
fn unusable_prompts_fall_back_to_default_with_real_tokenizer() {
    let Some((mut default_lm, _)) = lm_or_skip() else {
        return;
    };
    let default = upgraded(&mut default_lm, b"", Duration::from_secs(15));
    drop(default_lm);
    // The first prompt is under the byte limit but exceeds the token limit.
    for system_prompt in ["word ".repeat(1500), "\0".to_owned(), " \n\t".to_owned()] {
        let cfg = LlamaConfig {
            system_prompt,
            ..LlamaConfig::default()
        };
        let (mut lm, _) = lm_with_config_or_skip(cfg).unwrap();
        let actual = upgraded(&mut lm, b"", Duration::from_secs(15));
        assert!(
            total_variation(&default, &actual) < 1e-4,
            "invalid prompt must restore default conditioning"
        );
    }
}

//! keyb-lm-llama: llama.cpp-backed character model (SPEC §6).
//!
//! `LlamaLm` satisfies `keyb_core::LanguageModel` synchronously: a cache hit
//! answers with the real (smoothed) LLM distribution; a miss answers with PPM
//! alone and enqueues work on the inference worker. When the worker's answer
//! lands, `poll_upgrades` hands it to the engine, which animates box sizes to
//! the upgraded partition (SPEC §6.3) — boxes never pop.

mod marginal;
mod worker;

pub use worker::{FromWorker, ToWorker};

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crossbeam_channel::{Receiver, Sender, TrySendError};
use keyb_core::{Alphabet, LanguageModel};
use keyb_lm_ppm::PpmModel;

#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub model_path: PathBuf,
    pub n_threads: i32,
    /// Total KV cells shared by all branches (unified cache).
    pub n_ctx: u32,
    /// Live KV branches beyond the pinned preamble (SPEC §6.5).
    pub max_branches: usize,
    /// PPM share in the blend: p = (1-λ)·llm + λ·ppm (SPEC §6.3).
    pub lambda: f64,
    /// Per-symbol probability floor after blending.
    pub p_min: f64,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        LlamaConfig {
            model_path: default_model_path(),
            n_threads: (std::thread::available_parallelism().map_or(4, |n| n.get()) / 2).clamp(2, 8)
                as i32,
            n_ctx: 4096,
            max_branches: 32,
            lambda: 0.05,
            p_min: 1e-4,
        }
    }
}

/// `$XDG_DATA_HOME/keyb/models/<default model>` (SPEC §9).
pub fn default_model_path() -> PathBuf {
    let data = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_default();
    data.join("keyb/models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
}

pub struct LlamaLm {
    ppm: PpmModel,
    n_sym: usize,
    lambda: f64,
    p_min: f64,
    /// Raw LLM char distributions by context. Blending with PPM happens at
    /// serve time, so PPM's online learning is never served stale.
    cache: HashMap<Vec<u8>, Vec<f64>>,
    /// Contexts served with fallback sizing, awaiting a worker answer: these
    /// are the ones the engine must be told to re-partition. Every entry has
    /// a request in flight — the worker replies exactly once per request
    /// (even on failure), so entries can never strand.
    pending: HashSet<Vec<u8>>,
    tx: Sender<ToWorker>,
    rx: Receiver<FromWorker>,
    last_hint: Option<Vec<u8>>,
    worker_gone: bool,
    /// Context generation (SPEC §6.8): bumped per accepted `set_context`,
    /// carried in `ToWorker::SetContext` so a shed delivery cannot
    /// desynchronize the worker. Replies stamped with any other generation
    /// were computed under a replaced context block and are dropped.
    gen: u64,
    /// The context block last handed to the worker (`None` = preamble only).
    last_context: Option<String>,
    /// A `SetContext` that did not fit the channel, retried from
    /// `poll_upgrades`. Latest-wins: an undelivered older block is worthless.
    context_outbox: Option<Option<String>>,
    _worker: std::thread::JoinHandle<()>,
}

#[derive(Debug)]
pub struct LlamaLmError(pub String);

impl std::fmt::Display for LlamaLmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "keyb-lm-llama: {}", self.0)
    }
}

impl std::error::Error for LlamaLmError {}

impl LlamaLm {
    /// Spawns the inference worker and blocks until the model is loaded and
    /// the preamble branch is warm (cold-start budget: SPEC §10).
    pub fn new(alphabet: &Alphabet, cfg: LlamaConfig) -> Result<Self, LlamaLmError> {
        let (tx, worker_rx) = crossbeam_channel::bounded::<ToWorker>(512);
        let (worker_tx, rx) = crossbeam_channel::bounded::<FromWorker>(512);
        let (ready_tx, ready_rx) = crossbeam_channel::bounded(1);
        let (lambda, p_min) = (cfg.lambda, cfg.p_min);
        let a = alphabet.clone();
        let handle = std::thread::Builder::new()
            .name("keyb-llama".into())
            .spawn(move || worker::run(cfg, a, worker_rx, worker_tx, ready_tx))
            .map_err(|e| LlamaLmError(format!("spawn: {e}")))?;
        ready_rx
            .recv()
            .map_err(|_| LlamaLmError("worker died during startup".into()))?
            .map_err(LlamaLmError)?;
        Ok(LlamaLm {
            ppm: PpmModel::seeded(alphabet),
            n_sym: alphabet.len(),
            lambda,
            p_min,
            cache: HashMap::new(),
            pending: HashSet::new(),
            tx,
            rx,
            last_hint: None,
            worker_gone: false,
            gen: 0,
            last_context: None,
            context_outbox: None,
            _worker: handle,
        })
    }

    /// SPEC §6.3: p = (1-λ)·llm + λ·ppm, floored at p_min, renormalized.
    /// An all-zero LLM answer (everything out-of-alphabet) degrades to PPM.
    fn blend(&mut self, ctx: &[u8], llm: &[f64]) -> Vec<f64> {
        let ppm = self.ppm.predict(ctx);
        let llm_sum: f64 = llm.iter().sum();
        let lambda = if llm_sum > 0.0 { self.lambda } else { 1.0 };
        let mut p: Vec<f64> = llm
            .iter()
            .zip(&ppm)
            .map(|(&l, &q)| ((1.0 - lambda) * l + lambda * q).max(self.p_min))
            .collect();
        let sum: f64 = p.iter().sum();
        for v in p.iter_mut() {
            *v /= sum;
        }
        p
    }

    /// Try to hand `msg` to the worker without ever stalling the frame loop.
    /// Returns whether the message was accepted: callers must not record
    /// state that assumes delivery (in-flight markers, hint dedup) when it
    /// wasn't, or a shed message strands that state forever.
    fn send(&mut self, msg: ToWorker) -> bool {
        match self.tx.try_send(msg) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => false, // shed load; caller retries later
            Err(TrySendError::Disconnected(_)) => {
                if !self.worker_gone {
                    self.worker_gone = true;
                    eprintln!(
                        "keyb-lm-llama: inference worker gone — cached + PPM only from here on"
                    );
                }
                false
            }
        }
    }
}

impl LanguageModel for LlamaLm {
    fn predict(&mut self, context: &[u8]) -> Vec<f64> {
        if let Some(raw) = self.cache.get(context).cloned() {
            return self.blend(context, &raw);
        }
        // Mark in-flight only when the request actually reached the queue: a
        // shed request must stay re-requestable by the next expansion.
        if !self.pending.contains(context)
            && self.send(ToWorker::Predict {
                ctx: context.to_vec(),
            })
        {
            self.pending.insert(context.to_vec());
        }
        // Deadline missed by definition (nothing cached): PPM alone (§6.3).
        self.ppm.predict(context)
    }

    fn poll_upgrades(&mut self) -> Vec<(Vec<u8>, Vec<f64>)> {
        // Retry an undelivered context change before accepting answers, so
        // the generation gate below validates against the newest state.
        if let Some(text) = self.context_outbox.take() {
            if !self.send(ToWorker::SetContext {
                text: text.clone(),
                gen: self.gen,
            }) {
                self.context_outbox = Some(text);
            }
        }
        let mut out = Vec::new();
        while let Ok(FromWorker { ctx, gen, dist }) = self.rx.try_recv() {
            // Stale era: computed under a replaced context block. Markers
            // were all cleared when the change was accepted, so a stale
            // reply must not touch `pending` — it could belong to a
            // re-issued request that is still in flight.
            if gen != self.gen {
                continue;
            }
            let Some(raw) = dist else {
                // Not serviced (compute failed or the queue shed it): clear
                // the in-flight marker so a later predict re-requests. The
                // node keeps fallback sizing meanwhile.
                self.pending.remove(&ctx);
                continue;
            };
            if self.cache.len() > 16384 {
                // Shed half rather than wiping: a full clear reverts every
                // box on screen to fallback sizing at once mid-session.
                let mut keep = false;
                self.cache.retain(|_, _| {
                    keep = !keep;
                    keep
                });
            }
            self.cache.insert(ctx.clone(), raw.clone());
            // Only contexts that were actually served with fallback need
            // re-partitioning; speculative answers just warm the cache.
            if self.pending.remove(&ctx) {
                let blended = self.blend(&ctx, &raw);
                out.push((ctx, blended));
            }
        }
        out
    }

    fn frame_hint(&mut self, center_context: &[u8]) {
        if self.last_hint.as_deref() == Some(center_context) {
            return;
        }
        if self.send(ToWorker::Hint {
            ctx: center_context.to_vec(),
        }) {
            self.last_hint = Some(center_context.to_vec());
        }
    }

    /// SPEC §6.8: hand a new context block to the worker and invalidate
    /// everything derived from the old one. In-flight markers are cleared
    /// wholesale: their requests will be answered (exactly-once) with a
    /// stale generation and dropped, while the engine immediately
    /// re-requests the visible tree under the new generation.
    fn set_context(&mut self, context: Option<&str>) {
        let context = context.filter(|t| !t.trim().is_empty());
        if self.last_context.as_deref() == context {
            return;
        }
        self.last_context = context.map(str::to_string);
        self.gen += 1;
        self.cache.clear();
        self.pending.clear();
        if self.send(ToWorker::SetContext {
            text: self.last_context.clone(),
            gen: self.gen,
        }) {
            // A later successful delivery must not let poll_upgrades replay a
            // previously shed older block under this generation.
            self.context_outbox = None;
        } else {
            self.context_outbox = Some(self.last_context.clone());
        }
    }

    fn release(&mut self, context: &[u8]) {
        if self.pending.remove(context) {
            self.send(ToWorker::Cancel {
                ctx: context.to_vec(),
            });
        }
    }

    fn learn(&mut self, context: &[u8], sym: u8) {
        self.ppm.learn(context, sym);
    }

    fn unlearn(&mut self, context: &[u8], sym: u8) {
        self.ppm.unlearn(context, sym);
    }

    fn alphabet_len(&self) -> usize {
        self.n_sym
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::{bounded, Receiver};

    /// A `LlamaLm` wired to test-controlled channel ends instead of the real
    /// inference worker, so the generation protocol (SPEC §6.8) can be
    /// exercised without a model.
    fn test_lm() -> (
        LlamaLm,
        Receiver<ToWorker>,
        crossbeam_channel::Sender<FromWorker>,
    ) {
        let alphabet = Alphabet::english_v1();
        let n_sym = alphabet.len();
        let (tx, worker_rx) = bounded::<ToWorker>(512);
        let (worker_tx, rx) = bounded::<FromWorker>(512);
        let lm = LlamaLm {
            ppm: PpmModel::seeded(&alphabet),
            n_sym,
            lambda: 0.05,
            p_min: 1e-4,
            cache: HashMap::new(),
            pending: HashSet::new(),
            tx,
            rx,
            last_hint: None,
            worker_gone: false,
            gen: 0,
            last_context: None,
            context_outbox: None,
            _worker: std::thread::spawn(|| {}),
        };
        (lm, worker_rx, worker_tx)
    }

    #[test]
    fn set_context_sends_generation_and_invalidates() {
        let (mut lm, worker_rx, _wtx) = test_lm();
        // Pre-populate cache and an in-flight marker: both belong to the old
        // generation and must be invalidated by the change.
        lm.cache
            .insert(b"ab".to_vec(), vec![1.0 / lm.n_sym as f64; lm.n_sym]);
        lm.pending.insert(b"ab".to_vec());

        lm.set_context(Some("alice: see you thursday?"));

        let msg = worker_rx.try_recv().expect("SetContext must be sent");
        match msg {
            ToWorker::SetContext { text, gen } => {
                assert_eq!(text.as_deref(), Some("alice: see you thursday?"));
                assert_eq!(gen, 1, "first change bumps generation to 1");
            }
            other => panic!("expected SetContext, got {other:?}"),
        }
        assert!(lm.cache.is_empty(), "cache is old-context data");
        assert!(lm.pending.is_empty(), "markers are old-generation state");
        assert!(lm.context_outbox.is_none());
    }

    #[test]
    fn set_context_unchanged_is_noop() {
        let (mut lm, worker_rx, _wtx) = test_lm();
        lm.set_context(None);
        lm.set_context(Some("chat log"));
        lm.set_context(Some("chat log"));
        let mut sent = 0;
        while worker_rx.try_recv().is_ok() {
            sent += 1;
        }
        assert_eq!(sent, 1, "only the real change is sent");
        assert_eq!(lm.gen, 1);
    }

    #[test]
    fn stale_generation_replies_are_dropped() {
        let (mut lm, _worker_rx, worker_tx) = test_lm();
        lm.set_context(Some("context A")); // gen 1
                                           // A request issued under the new generation gets its marker.
        let ctx = b"hello".to_vec();
        let _ = lm.predict(&ctx);
        assert!(lm.pending.contains(&ctx));

        // A reply from before the change: dropped without touching markers.
        worker_tx
            .send(FromWorker {
                ctx: ctx.clone(),
                gen: 0,
                dist: Some(vec![1.0 / lm.n_sym as f64; lm.n_sym]),
            })
            .unwrap();
        assert!(lm.poll_upgrades().is_empty());
        assert!(
            lm.pending.contains(&ctx),
            "stale reply must not clear a new-generation marker"
        );
        assert!(lm.cache.is_empty(), "stale data must never be cached");

        // The current-generation answer lands normally.
        worker_tx
            .send(FromWorker {
                ctx: ctx.clone(),
                gen: 1,
                dist: Some(vec![1.0 / lm.n_sym as f64; lm.n_sym]),
            })
            .unwrap();
        let upgrades = lm.poll_upgrades();
        assert_eq!(upgrades.len(), 1, "fresh reply upgrades the node");
        assert_eq!(upgrades[0].0, ctx);
        assert!(!lm.pending.contains(&ctx));
    }

    #[test]
    fn shed_set_context_retries_from_outbox() {
        let (mut lm, worker_rx, _wtx) = test_lm();
        // Fill the control channel so the SetContext send is shed.
        while lm.tx.try_send(ToWorker::Hint { ctx: vec![] }).is_ok() {}
        lm.set_context(Some("deferred context"));
        assert_eq!(
            lm.context_outbox,
            Some(Some("deferred context".to_string())),
            "shed SetContext parks in the outbox"
        );

        // The worker drains; the next poll_upgrades retries the delivery.
        while worker_rx.try_recv().is_ok() {}
        assert!(lm.poll_upgrades().is_empty());
        assert!(lm.context_outbox.is_none(), "outbox flushed");
        match worker_rx.try_recv() {
            Ok(ToWorker::SetContext { text, gen }) => {
                assert_eq!(text.as_deref(), Some("deferred context"));
                assert_eq!(gen, lm.gen, "carries the adapter generation");
            }
            other => panic!("expected SetContext, got {other:?}"),
        }
    }

    #[test]
    fn set_context_empty_or_whitespace_is_noop() {
        let (mut lm, worker_rx, _wtx) = test_lm();
        lm.set_context(Some(""));
        lm.set_context(Some("  \n\t"));
        lm.set_context(None);
        assert!(
            worker_rx.try_recv().is_err(),
            "blank context is preamble-only, same as the initial state"
        );
        assert_eq!(lm.gen, 0);
        assert!(lm.last_context.is_none());
    }

    #[test]
    fn set_context_none_clears_an_active_block() {
        let (mut lm, worker_rx, _wtx) = test_lm();
        lm.set_context(Some("chat log"));
        assert!(worker_rx.try_recv().is_ok());
        lm.set_context(None);
        match worker_rx.try_recv() {
            Ok(ToWorker::SetContext { text, gen }) => {
                assert_eq!(text, None);
                assert_eq!(gen, 2);
            }
            other => panic!("expected clearing SetContext, got {other:?}"),
        }
        assert!(lm.last_context.is_none());
        assert!(lm.cache.is_empty());
        assert!(lm.pending.is_empty());
    }

    #[test]
    fn shed_set_context_latest_wins_in_outbox() {
        let (mut lm, _worker_rx, _wtx) = test_lm();
        while lm.tx.try_send(ToWorker::Hint { ctx: vec![] }).is_ok() {}
        lm.set_context(Some("first"));
        lm.set_context(Some("second"));
        assert_eq!(lm.gen, 2);
        assert_eq!(
            lm.context_outbox,
            Some(Some("second".to_string())),
            "an undelivered older block is worthless"
        );
    }

    #[test]
    fn successful_set_context_drops_a_shed_older_outbox() {
        let (mut lm, worker_rx, _wtx) = test_lm();
        while lm.tx.try_send(ToWorker::Hint { ctx: vec![] }).is_ok() {}
        lm.set_context(Some("first"));
        assert_eq!(
            lm.context_outbox,
            Some(Some("first".to_string())),
            "first change is shed"
        );

        // Worker drains; the next change must deliver and forget the outbox,
        // otherwise poll_upgrades would replay "first" under generation 2.
        while worker_rx.try_recv().is_ok() {}
        lm.set_context(Some("second"));
        assert!(
            lm.context_outbox.is_none(),
            "delivered change must not leave a stale outbox: {:?}",
            lm.context_outbox
        );
        match worker_rx.try_recv() {
            Ok(ToWorker::SetContext { text, gen }) => {
                assert_eq!(text.as_deref(), Some("second"));
                assert_eq!(gen, 2);
            }
            other => panic!("expected SetContext(second), got {other:?}"),
        }
        assert!(worker_rx.try_recv().is_err(), "no extra SetContext queued");
        assert!(lm.poll_upgrades().is_empty());
        assert!(
            worker_rx.try_recv().is_err(),
            "poll_upgrades must not replay the shed first block"
        );
    }
}

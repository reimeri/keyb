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
                    eprintln!("keyb-lm-llama: inference worker gone — cached + PPM only from here on");
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
        let mut out = Vec::new();
        while let Ok(FromWorker { ctx, dist }) = self.rx.try_recv() {
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

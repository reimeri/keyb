//! Inference worker (SPEC §6.4–6.6): one thread owns the llama.cpp context.
//!
//! Requests are character contexts (absolute alphabet symbols). The worker
//! keeps a tree of KV-cache branches keyed by token sequence: expanding a
//! deeper context copies the longest cached prefix (copy-on-write cells in
//! the unified cache) and decodes only the divergent tail. Each serviced
//! branch stores its full next-token distribution so every intra-token
//! request against it is answered without another forward pass.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender};
use keyb_core::Alphabet;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
// `token_to_bytes` is deprecated in favor of raw `token_to_piece_bytes`, but
// it is exactly the buffer-retrying wrapper we want for vocab preprocessing.
#[allow(deprecated)]
use llama_cpp_2::model::Special;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use crate::marginal::VocabIndex;
use crate::{
    LlamaConfig, DEFAULT_SYSTEM_PROMPT, MAX_SYSTEM_PROMPT_BYTES, MAX_SYSTEM_PROMPT_TOKENS,
};

#[derive(Debug)]
pub enum ToWorker {
    /// The engine expanded a node with fallback sizing; compute the real
    /// distribution for this context.
    Predict { ctx: Vec<u8> },
    /// Crosshair moved: the deepest expanded node under the view center.
    /// Drives prioritization and speculative expansion.
    Hint { ctx: Vec<u8> },
    /// The node was evicted; drop pending work for this context.
    Cancel { ctx: Vec<u8> },
    /// Replace the pinned context block (SPEC §6.8): text the user is
    /// replying to, decoded into the prefix between the preamble and the
    /// committed text. `None` restores the preamble-only prefix. `gen` is
    /// the adapter's generation counter — carried in the message (rather
    /// than incremented here) so a shed-and-retried delivery can never
    /// desynchronize the two sides. Every queued request is answered "not
    /// serviced" under the old generation; all branches and computed
    /// distributions are then flushed.
    SetContext { text: Option<String>, gen: u64 },
}

pub struct FromWorker {
    pub ctx: Vec<u8>,
    /// Context generation this answer belongs to (SPEC §6.8): the worker's
    /// generation counter at reply time, bumped by every `SetContext`.
    /// Answers from before the adapter's current generation are stale —
    /// computed under a replaced context block — and must be dropped
    /// without touching in-flight markers.
    pub gen: u64,
    /// Raw LLM char distribution over the alphabet (pre-smoothing). Sums to 1
    /// unless everything was out-of-alphabet (then it is all zeros and the
    /// adapter keeps its fallback). `None` means the request was not serviced
    /// (compute failed, or the queue shed it): the adapter clears its
    /// in-flight marker so a later `predict` may re-request. Every `Predict`
    /// gets exactly one reply — a silently dropped request would strand its
    /// context on fallback sizing forever.
    pub dist: Option<Vec<f64>>,
}

/// A skipped rebuild still has to adopt the adapter's generation: it may
/// have advanced through context changes that were shed before reaching us.
fn same_context_adopt_generation(
    current: &Option<String>,
    current_gen: &mut u64,
    text: &Option<String>,
    gen: u64,
) -> bool {
    if current != text {
        return false;
    }
    *current_gen = gen;
    true
}

/// How much of the healed token mass may end exactly at the current prefix
/// before we pay one extra forward pass to resolve its continuation (§6.2).
const EXACT_MASS_THRESHOLD: f64 = 0.05;
/// Speculative fan-out (SPEC §6.6): top-k next symbols of the hinted context.
const SPECULATE_K: usize = 4;
/// How far past the crosshair the argmax chain is speculated while idle.
const SPECULATE_DEPTH: usize = 6;
/// Queue bound: beyond this the stalest requests are shed with an explicit
/// "not serviced" reply (never silently — see `FromWorker::dist`).
const MAX_PENDING: usize = 256;
/// LLM context window over committed symbols. The window start moves in
/// `WINDOW_STEP` jumps so branch prefixes stay shareable between jumps.
const WINDOW: usize = 800;
const WINDOW_STEP: usize = 256;

/// Framing for the captured context block (SPEC §6.8). The block sits in the
/// pinned prefix between the preamble and the committed text, so the user's
/// text continues naturally from what they are replying to. Kept deliberately
/// plain — the small default model follows a transcript far better than a
/// chat template.
const CONTEXT_HEADER: &str = "The writer is replying to the following text:\n";
const CONTEXT_TAIL: &str = "\n\nThe writer's reply:\n";
/// Token budget for the full framed context block. The body is tail-trimmed
/// so the most recent lines survive. Reserved separately from the static prompt.
const CONTEXT_MAX_TOKENS: usize = 1024;

/// Reserve a byte-token worst case for the 800-symbol window plus one healed
/// continuation and some headroom. The static prompt can never take this space
/// or the captured-context block's budget.
fn prompt_token_budget(n_ctx: u32, max_symbol_bytes: usize) -> usize {
    let reserved = CONTEXT_MAX_TOKENS + WINDOW * max_symbol_bytes + 32;
    MAX_SYSTEM_PROMPT_TOKENS.min((n_ctx as usize).saturating_sub(reserved))
}

fn validated_prompt_tokens(
    prompt: &str,
    budget: usize,
    tokenize: impl Fn(&str) -> Result<Vec<LlamaToken>, String>,
) -> Result<Vec<LlamaToken>, String> {
    if prompt.trim().is_empty() {
        return Err("prompt is empty".into());
    }
    if prompt.len() > MAX_SYSTEM_PROMPT_BYTES {
        return Err(format!("prompt exceeds {MAX_SYSTEM_PROMPT_BYTES} bytes"));
    }
    // Preserve user content; only add missing separating newlines. In
    // particular, the default prompt keeps exactly its existing tokenization.
    let mut framed = prompt.to_owned();
    if !framed.ends_with('\n') {
        framed.push('\n');
    }
    if !framed.ends_with("\n\n") {
        framed.push('\n');
    }
    let tokens = tokenize(&framed)?;
    if tokens.is_empty() || tokens.len() > budget {
        return Err(format!(
            "prompt has {} tokens; allowed range is 1–{budget}",
            tokens.len()
        ));
    }
    Ok(tokens)
}

/// The static prompt is already validated and tokenized, and is reused on
/// every context change. Only the captured body is ever tail-trimmed.
fn assemble_prefix(
    prompt: &[LlamaToken],
    context: Option<&str>,
    tokenize: impl Fn(&str) -> Result<Vec<LlamaToken>, String>,
) -> Result<Vec<LlamaToken>, String> {
    let mut tokens = prompt.to_vec();
    if let Some(context) = context {
        let head = tokenize(CONTEXT_HEADER)?;
        let mut body = tokenize(context)?;
        let tail = tokenize(CONTEXT_TAIL)?;
        let budget = CONTEXT_MAX_TOKENS.saturating_sub(head.len() + tail.len());
        if body.len() > budget {
            body.drain(..body.len() - budget);
        }
        tokens.extend(head);
        tokens.extend(body);
        tokens.extend(tail);
    }
    Ok(tokens)
}

/// Decode a sequence in logical-batch-sized chunks, using absolute positions.
/// Only the final token needs logits. This also handles long captured blocks
/// on backends whose n_batch is smaller than the total pinned prefix.
fn decode_tokens(
    ctx: &mut LlamaContext<'_>,
    tokens: &[LlamaToken],
    start: i32,
    seq: i32,
) -> Result<Arc<[f32]>, String> {
    if tokens.is_empty() {
        return Err("cannot decode an empty sequence".into());
    }
    let batch_size = (ctx.n_batch() as usize).max(1);
    for (chunk_index, chunk) in tokens.chunks(batch_size).enumerate() {
        let offset = chunk_index * batch_size;
        let mut batch = LlamaBatch::new(chunk.len(), 1);
        for (i, &token) in chunk.iter().enumerate() {
            batch
                .add(
                    token,
                    start + (offset + i) as i32,
                    &[seq],
                    offset + i + 1 == tokens.len(),
                )
                .map_err(|e| format!("batch add: {e}"))?;
        }
        ctx.decode(&mut batch).map_err(|e| format!("decode: {e}"))?;
    }
    let last_index = ((tokens.len() - 1) % batch_size) as i32;
    Ok(softmax(ctx.get_logits_ith(last_index)).into())
}

struct Branch {
    seq: i32,
    /// softmax over the whole vocab at this branch point.
    probs: Arc<[f32]>,
    last_used: u64,
}

struct Worker<'m> {
    model: &'m LlamaModel,
    ctx: LlamaContext<'m>,
    vocab: VocabIndex,
    alphabet: Alphabet,
    /// Length of the pinned prefix (preamble + optional context block) —
    /// the position offset of every user-text token.
    n_preamble: i32,
    /// Full static prompt, including separator, reused across context rebuilds.
    system_prompt_tokens: Vec<LlamaToken>,
    /// The captured context block as last accepted (`None` = preamble only).
    context_text: Option<String>,
    /// Context generation: bumped per accepted `SetContext`, stamped onto
    /// every reply so the adapter can drop stale answers.
    gen: u64,
    /// Branch budget beyond the pinned prefix (from config); the free list
    /// is rebuilt to this after a context change flushes all branches.
    max_branches: usize,
    /// Branches keyed by user-text token sequence (prefix implicit).
    /// The empty key (prefix only, seq 0) is pinned.
    branches: HashMap<Box<[LlamaToken]>, Branch>,
    free_seqs: Vec<i32>,
    tick: u64,
    /// Char distributions already computed this session, so speculation and
    /// duplicate requests cost nothing.
    computed: HashMap<Vec<u8>, Vec<f64>>,
}

impl<'m> Worker<'m> {
    fn touch(&mut self, key: &[LlamaToken]) {
        self.tick += 1;
        if let Some(b) = self.branches.get_mut(key) {
            b.last_used = self.tick;
        }
    }

    /// Tokenize the pinned prefix: the preamble, then — when a context block
    /// is set — header + tail-trimmed body + tail (SPEC §6.8). The parts are
    /// tokenized separately so the budget trims exactly the body; the
    /// boundaries fall on newlines, where BPE merging is a no-op in practice.
    fn prefix_tokens(&self) -> Result<Vec<LlamaToken>, String> {
        assemble_prefix(
            &self.system_prompt_tokens,
            self.context_text.as_deref(),
            |text| {
                self.model
                    .str_to_token(text, AddBos::Never)
                    .map_err(|e| format!("prefix tokenize: {e}"))
            },
        )
    }

    /// Decode the pinned prefix into sequence 0. Release all old KV sequences
    /// first: otherwise branches sharing the old prefix can exhaust capacity
    /// while the new prefix is decoded. Callers must invalidate branch metadata
    /// immediately afterwards, before servicing any new prediction.
    fn rebuild_prefix(&mut self) -> Result<(), String> {
        let toks = self.prefix_tokens()?;
        self.ctx
            .clear_kv_cache_seq(None, None, None)
            .map_err(|e| format!("kv clear: {e}"))?;
        let probs = decode_tokens(&mut self.ctx, &toks, 0, 0)?;
        self.n_preamble = toks.len() as i32;
        self.tick += 1;
        self.branches.insert(
            Vec::new().into_boxed_slice(),
            Branch {
                seq: 0,
                probs,
                last_used: self.tick,
            },
        );
        Ok(())
    }

    /// Sequence id for `tokens`, creating the branch (prefix copy + decode of
    /// the tail) if needed. Returns the branch's next-token distribution.
    fn ensure_branch(&mut self, tokens: &[LlamaToken]) -> Result<Arc<[f32]>, String> {
        if self.branches.contains_key(tokens) {
            self.touch(tokens);
            return Ok(self.branches[tokens].probs.clone());
        }
        // Longest cached prefix is the copy source (empty key always hits).
        let mut k = tokens.len() - 1;
        while k > 0 && !self.branches.contains_key(&tokens[..k]) {
            k -= 1;
        }
        let src_seq = self.branches[&tokens[..k]].seq;
        let seq = match self.free_seqs.pop() {
            Some(s) => s,
            None => self.evict_lru(src_seq)?,
        };
        // From here this call owns `seq`: any failure must hand it back to
        // `free_seqs`, or the branch budget shrinks by one forever and the
        // worker eventually runs out of sequences entirely.
        match self.decode_branch(seq, src_seq, tokens, k) {
            Ok(probs) => {
                self.tick += 1;
                self.branches.insert(
                    tokens.into(),
                    Branch {
                        seq,
                        probs: probs.clone(),
                        last_used: self.tick,
                    },
                );
                Ok(probs)
            }
            Err(e) => {
                self.free_seqs.push(seq);
                Err(e)
            }
        }
    }

    /// Fill `seq` with the KV state for `tokens` (prefix copied from
    /// `src_seq`, tail decoded) and return the next-token distribution.
    fn decode_branch(
        &mut self,
        seq: i32,
        src_seq: i32,
        tokens: &[LlamaToken],
        k: usize,
    ) -> Result<Arc<[f32]>, String> {
        self.ctx
            .clear_kv_cache_seq(Some(seq as u32), None, None)
            .map_err(|e| format!("kv clear: {e}"))?;
        self.ctx
            .copy_kv_cache_seq(
                src_seq,
                seq,
                Some(0),
                Some((self.n_preamble + k as i32) as u32),
            )
            .map_err(|e| format!("kv copy: {e}"))?;
        decode_tokens(&mut self.ctx, &tokens[k..], self.n_preamble + k as i32, seq)
    }

    /// Free the least-recently-used branch (never the pinned preamble at the
    /// empty key, never `keep`) and hand back its sequence id.
    fn evict_lru(&mut self, keep: i32) -> Result<i32, String> {
        let victim = self
            .branches
            .iter()
            .filter(|(k, b)| !k.is_empty() && b.seq != keep)
            .min_by_key(|(_, b)| b.last_used)
            .map(|(k, _)| k.clone())
            .ok_or_else(|| "no evictable branch".to_string())?;
        let b = self.branches.remove(&victim).expect("victim exists");
        Ok(b.seq)
    }

    /// Committed-symbol context → windowed text the model sees.
    fn window_text(&self, syms: &[u8]) -> String {
        let start = if syms.len() <= WINDOW {
            0
        } else {
            (syms.len() - WINDOW).div_ceil(WINDOW_STEP) * WINDOW_STEP
        };
        self.alphabet.decode(&syms[start..])
    }

    /// P(next char | ctx) from the LLM alone (SPEC §6.2): roll back to the
    /// earliest token boundary that can heal the current suffix, then resolve
    /// significant exact-token mass through those tokens' actual branches.
    fn compute(&mut self, ctx: &[u8]) -> Result<Vec<f64>, String> {
        if let Some(d) = self.computed.get(ctx) {
            return Ok(d.clone());
        }
        let text = self.window_text(ctx);
        let tokens = if text.is_empty() {
            Vec::new()
        } else {
            self.model
                .str_to_token(&text, AddBos::Never)
                .map_err(|e| format!("tokenize: {e}"))?
        };
        let token_ids: Vec<u32> = tokens.iter().map(|t| t.0 as u32).collect();
        let (split, s) = self.vocab.healing_suffix(&token_ids);
        let prefix_toks = &tokens[..split];
        let probs_a = self.ensure_branch2(prefix_toks)?;
        let (mut mass, exact) = self.vocab.char_masses(&probs_a, &s);
        let cont: f64 = mass.iter().sum();
        if exact > EXACT_MASS_THRESHOLD * (exact + cont) && !tokens.is_empty() {
            for exact_id in self.vocab.exact_tokens(&s) {
                let token_mass = probs_a[exact_id as usize] as f64;
                if token_mass == 0.0 {
                    continue;
                }
                let mut exact_branch = prefix_toks.to_vec();
                exact_branch.push(LlamaToken(exact_id as i32));
                let probs_b = self.ensure_branch2(&exact_branch)?;
                let (root, _) = self.vocab.char_masses(&probs_b, &[]);
                let root_sum: f64 = root.iter().sum();
                if root_sum > 0.0 {
                    for (m, r) in mass.iter_mut().zip(&root) {
                        *m += token_mass * r / root_sum;
                    }
                }
            }
        }
        let sum: f64 = mass.iter().sum();
        if sum > 0.0 {
            for m in mass.iter_mut() {
                *m /= sum;
            }
        }
        self.computed.insert(ctx.to_vec(), mass.clone());
        if self.computed.len() > 16384 {
            // Shed half rather than wiping, so a long session degrades
            // gradually instead of recomputing everything at once.
            let mut keep = false;
            self.computed.retain(|_, _| {
                keep = !keep;
                keep
            });
        }
        Ok(mass)
    }

    /// `ensure_branch` for possibly-empty token slices (empty = preamble).
    fn ensure_branch2(&mut self, tokens: &[LlamaToken]) -> Result<Arc<[f32]>, String> {
        if tokens.is_empty() {
            self.touch(&[]);
            return Ok(self.branches[&[] as &[LlamaToken]].probs.clone());
        }
        self.ensure_branch(tokens)
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    for v in out.iter_mut() {
        *v /= sum;
    }
    out
}

// Shared by production workers and optional real-model unit tests. llama.cpp
// may initialize its backend only once per process.
fn inference_backend() -> Result<&'static LlamaBackend, String> {
    static BACKEND: std::sync::OnceLock<Result<LlamaBackend, String>> = std::sync::OnceLock::new();
    BACKEND
        .get_or_init(|| {
            llama_cpp_2::send_logs_to_tracing(
                llama_cpp_2::LogOptions::default().with_logs_enabled(false),
            );
            LlamaBackend::init().map_err(|e| format!("backend: {e}"))
        })
        .as_ref()
        .map_err(Clone::clone)
}

pub fn run(
    cfg: LlamaConfig,
    alphabet: Alphabet,
    rx: Receiver<ToWorker>,
    tx: Sender<FromWorker>,
    ready: Sender<Result<(), String>>,
) {
    let setup = || -> Result<(&'static LlamaBackend, LlamaModel), String> {
        let backend = inference_backend()?;
        let params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(backend, &cfg.model_path, &params)
            .map_err(|e| format!("load {}: {e}", cfg.model_path.display()))?;
        Ok((backend, model))
    };
    let (backend, model) = match setup() {
        Ok(x) => x,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let n_seq_max = cfg.max_branches as u32 + 8;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(std::num::NonZeroU32::new(cfg.n_ctx))
        .with_n_seq_max(n_seq_max)
        .with_kv_unified(true)
        .with_n_threads(cfg.n_threads)
        .with_n_threads_batch(cfg.n_threads)
        .with_no_perf(true);
    let ctx = match model.new_context(backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            let _ = ready.send(Err(format!("context: {e}")));
            return;
        }
    };

    let n_vocab = model.n_vocab() as u32;
    #[allow(deprecated)]
    let vocab = VocabIndex::new(
        (0..n_vocab).map(|t| {
            model
                .token_to_bytes(LlamaToken(t as i32), Special::Plaintext)
                .unwrap_or_default()
        }),
        &alphabet,
    );

    let max_symbol_bytes = (0..alphabet.len())
        .map(|symbol| alphabet.char_of(symbol as u8).len_utf8())
        .max()
        .unwrap_or(1);
    let budget = prompt_token_budget(ctx.n_ctx(), max_symbol_bytes);
    let tokenize = |text: &str| {
        model
            .str_to_token(text, AddBos::Never)
            .map_err(|e| format!("prompt tokenize: {e}"))
    };
    let system_prompt_tokens = match validated_prompt_tokens(&cfg.system_prompt, budget, tokenize) {
        Ok(tokens) => tokens,
        Err(error) => {
            eprintln!("keyb-lm-llama: instructions.txt system prompt: {error} — using default prompt; file left unchanged");
            match validated_prompt_tokens(DEFAULT_SYSTEM_PROMPT, budget, tokenize) {
                Ok(tokens) => tokens,
                Err(error) => {
                    let _ = ready.send(Err(format!("default prompt cannot fit context: {error}")));
                    return;
                }
            }
        }
    };

    let mut w = Worker {
        model: &model,
        ctx,
        vocab,
        alphabet,
        n_preamble: 0,
        system_prompt_tokens,
        context_text: None,
        gen: 0,
        max_branches: cfg.max_branches,
        branches: HashMap::new(),
        free_seqs: (1..cfg.max_branches as i32 + 1).rev().collect(),
        tick: 0,
        computed: HashMap::new(),
    };

    // Warm the pinned prefix branch (seq 0) so the first prediction is one
    // prefix-copy + short decode away. No context block exists at startup.
    if let Err(e) = w.rebuild_prefix() {
        let _ = ready.send(Err(e));
        return;
    }
    let _ = ready.send(Ok(()));

    // Request state: newest-first, hint-adjacent boosted, cancellable.
    let mut pending: Vec<Vec<u8>> = Vec::new();
    let mut hint: Vec<u8> = Vec::new();

    'main: loop {
        // Drain control traffic before every unit of work so cancellation and
        // re-prioritization act between forward passes, not behind them.
        loop {
            let msg = if pending.is_empty() {
                // Idle apart from speculation; poll gently.
                match rx.recv_timeout(Duration::from_millis(5)) {
                    Ok(m) => Some(m),
                    Err(crossbeam_channel::RecvTimeoutError::Timeout) => None,
                    Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break 'main,
                }
            } else {
                match rx.try_recv() {
                    Ok(m) => Some(m),
                    Err(crossbeam_channel::TryRecvError::Empty) => None,
                    Err(crossbeam_channel::TryRecvError::Disconnected) => break 'main,
                }
            };
            match msg {
                Some(ToWorker::Predict { ctx }) => {
                    if !pending.contains(&ctx) {
                        pending.push(ctx);
                    }
                    // Shed the stalest work rather than growing without
                    // bound; the reply lets the adapter re-request later.
                    while pending.len() > MAX_PENDING {
                        let old = pending.remove(0);
                        if tx
                            .send(FromWorker {
                                ctx: old,
                                gen: w.gen,
                                dist: None,
                            })
                            .is_err()
                        {
                            break 'main;
                        }
                    }
                }
                Some(ToWorker::Hint { ctx }) => hint = ctx,
                Some(ToWorker::Cancel { ctx }) => pending.retain(|c| c != &ctx),
                Some(ToWorker::SetContext { text, gen }) => {
                    let text = text.filter(|t| !t.trim().is_empty());
                    if same_context_adopt_generation(&w.context_text, &mut w.gen, &text, gen) {
                        continue;
                    }
                    w.context_text = text;
                    // Queued requests belong to the old generation: answer
                    // each "not serviced" stamped with it (preserving
                    // exactly-once) so the adapter drops the replies without
                    // clearing markers for re-issued requests.
                    for old in pending.drain(..) {
                        if tx
                            .send(FromWorker {
                                ctx: old,
                                gen: w.gen,
                                dist: None,
                            })
                            .is_err()
                        {
                            break 'main;
                        }
                    }
                    w.gen = gen;
                    if let Err(e) = w.rebuild_prefix() {
                        // No valid pinned branch remains after a partial
                        // decode. Exit so the adapter falls back to PPM rather
                        // than reusing logits from the replaced context.
                        eprintln!(
                            "keyb-lm-llama: context prefix rebuild: {e} — running on PPM only"
                        );
                        return;
                    }
                    // Every user-text branch shares prefix cells with seq 0
                    // and every computed distribution was conditioned on the
                    // old context: both are stale now.
                    w.branches.retain(|k, _| k.is_empty());
                    w.free_seqs = (1..w.max_branches as i32 + 1).rev().collect();
                    w.computed.clear();
                }
                None => break,
            }
        }

        // Pick: newest request on/near the hinted path, else newest.
        let pick = pending
            .iter()
            .rposition(|c| c.starts_with(&hint) || hint.starts_with(c))
            .or(if pending.is_empty() {
                None
            } else {
                Some(pending.len() - 1)
            });
        if let Some(i) = pick {
            let ctx = pending.remove(i);
            let reply = match w.compute(&ctx) {
                Ok(dist) => FromWorker {
                    ctx,
                    gen: w.gen,
                    dist: Some(dist),
                },
                Err(e) => {
                    eprintln!("keyb-lm-llama: {e}");
                    FromWorker {
                        ctx,
                        gen: w.gen,
                        dist: None,
                    }
                }
            };
            if tx.send(reply).is_err() {
                break 'main;
            }
            continue;
        }

        // Idle: speculate ahead of the crosshair (SPEC §6.6). Breadth-first
        // near the hint (top-k next symbols), then deepen along the argmax
        // chain — one forward pass per loop so real requests preempt.
        let mut cur = hint.clone();
        'speculate: for _ in 0..SPECULATE_DEPTH {
            let Some(dist) = w.computed.get(&cur).cloned() else {
                break;
            };
            let mut order: Vec<usize> = (0..dist.len()).collect();
            order.sort_by(|&a, &b| dist[b].total_cmp(&dist[a]));
            for &sym in order.iter().take(SPECULATE_K) {
                let mut child = cur.clone();
                child.push(sym as u8);
                if w.computed.contains_key(&child) {
                    continue;
                }
                match w.compute(&child) {
                    Ok(d) => {
                        if tx
                            .send(FromWorker {
                                ctx: child,
                                gen: w.gen,
                                dist: Some(d),
                            })
                            .is_err()
                        {
                            break 'main;
                        }
                    }
                    // Unsolicited work: no reply owed on failure.
                    Err(e) => eprintln!("keyb-lm-llama: speculate: {e}"),
                }
                break 'speculate; // one unit of work, then re-check traffic
            }
            cur.push(order[0] as u8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn byte_tokens(text: &str) -> Result<Vec<LlamaToken>, String> {
        Ok(text.bytes().map(|byte| LlamaToken(byte as i32)).collect())
    }

    #[test]
    fn default_config_preserves_the_original_prompt_exactly() {
        let cfg = LlamaConfig::default();
        assert_eq!(cfg.system_prompt, DEFAULT_SYSTEM_PROMPT);
        assert_eq!(
            validated_prompt_tokens(&cfg.system_prompt, 1024, byte_tokens).unwrap(),
            byte_tokens(DEFAULT_SYSTEM_PROMPT).unwrap()
        );
    }

    #[test]
    fn custom_prompt_is_not_prefixed_with_hidden_instructions() {
        for text in ["Use Finnish.", "Use Finnish.\n", "Use Finnish.\n\n"] {
            let tokens = validated_prompt_tokens(text, 1024, byte_tokens).unwrap();
            assert_eq!(tokens, byte_tokens("Use Finnish.\n\n").unwrap());
        }
        let text = "Use Finnish.  \n\n\n";
        assert_eq!(
            validated_prompt_tokens(text, 1024, byte_tokens).unwrap(),
            byte_tokens(text).unwrap()
        );
    }

    #[test]
    fn validation_rejects_empty_oversized_and_untokenizable_prompts() {
        for text in ["", " \t\n"] {
            assert!(validated_prompt_tokens(text, 1024, byte_tokens).is_err());
        }
        let oversized = "a".repeat(MAX_SYSTEM_PROMPT_BYTES + 1);
        assert!(validated_prompt_tokens(&oversized, 1024, |_| panic!(
            "must reject before tokenization"
        ))
        .is_err());
        // With this mock tokenizer the two separating newlines count too.
        assert!(validated_prompt_tokens("abcd", 5, byte_tokens).is_err());
        assert_eq!(
            validated_prompt_tokens("abcd", 6, byte_tokens)
                .unwrap()
                .len(),
            6
        );
        assert!(validated_prompt_tokens("a", 0, byte_tokens).is_err());
        assert!(validated_prompt_tokens("a", 1024, |_| Ok(vec![])).is_err());
        assert!(validated_prompt_tokens("a", 1024, |_| Err("tokenizer error".into())).is_err());
    }

    #[test]
    fn token_budget_reserves_context_and_worst_case_typed_window() {
        assert_eq!(prompt_token_budget(4096, 1), MAX_SYSTEM_PROMPT_TOKENS);
        assert_eq!(prompt_token_budget(2048, 1), 192);
        assert_eq!(prompt_token_budget(1024, 1), 0);
        assert_eq!(prompt_token_budget(4096, 4), 0);
    }

    #[test]
    fn context_changes_preserve_full_custom_prompt_and_clear_to_it() {
        let prompt = validated_prompt_tokens("My own instructions.", 1024, byte_tokens).unwrap();
        for context in [None, Some("first context"), Some("second context"), None] {
            let actual = assemble_prefix(&prompt, context, byte_tokens).unwrap();
            let mut expected = prompt.clone();
            if let Some(context) = context {
                expected.extend(
                    byte_tokens(&format!("{CONTEXT_HEADER}{context}{CONTEXT_TAIL}")).unwrap(),
                );
            }
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn long_context_trims_only_the_body_not_the_static_prompt() {
        let prompt = validated_prompt_tokens("My own instructions.", 1024, byte_tokens).unwrap();
        let context = format!("{}latest lines", "x".repeat(2048));
        let actual = assemble_prefix(&prompt, Some(&context), byte_tokens).unwrap();
        assert_eq!(&actual[..prompt.len()], &prompt);
        assert_eq!(actual.len(), prompt.len() + CONTEXT_MAX_TOKENS);
        let suffix = byte_tokens(&format!("latest lines{CONTEXT_TAIL}")).unwrap();
        assert!(actual.ends_with(&suffix));
    }

    /// Optional real-model regression: force a much smaller batch limit than
    /// the prefix and compare chunked decoding to explicit reference batches.
    /// Keep batch shapes identical: quantized kernels can produce measurably
    /// different probabilities for different matrix multiplication shapes.
    #[test]
    fn long_prefix_decodes_in_batches_with_correct_positions_and_logits() {
        let cfg = LlamaConfig::default();
        if !cfg.model_path.exists() {
            eprintln!("skipping: no model at {}", cfg.model_path.display());
            return;
        }
        let backend = inference_backend().unwrap();
        let model =
            LlamaModel::load_from_file(backend, &cfg.model_path, &LlamaModelParams::default())
                .unwrap();
        let params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(4096))
            .with_n_batch(32)
            .with_n_ubatch(32)
            .with_n_threads(2)
            .with_n_threads_batch(2);
        let mut ctx = model.new_context(backend, params).unwrap();
        let tokens = model
            .str_to_token(
                &"ordinary English text with punctuation. ".repeat(20),
                AddBos::Never,
            )
            .unwrap();
        assert!(tokens.len() > ctx.n_batch() as usize);
        let chunked = decode_tokens(&mut ctx, &tokens, 0, 0).unwrap();
        let reference_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(4096))
            .with_n_batch(32)
            .with_n_ubatch(32)
            .with_n_threads(2)
            .with_n_threads_batch(2);
        let mut reference_ctx = model.new_context(backend, reference_params).unwrap();
        let mut batch = LlamaBatch::new(32, 1);
        let mut last_index = 0;
        for (position, &token) in tokens.iter().enumerate() {
            last_index = batch.n_tokens();
            batch
                .add(token, position as i32, &[0], position + 1 == tokens.len())
                .unwrap();
            if batch.n_tokens() == 32 || position + 1 == tokens.len() {
                reference_ctx.decode(&mut batch).unwrap();
                batch.clear();
            }
        }
        let reference = softmax(reference_ctx.get_logits_ith(last_index));
        let distance: f64 = chunked
            .iter()
            .zip(reference)
            .map(|(a, b)| (*a as f64 - b as f64).abs())
            .sum();
        assert!(
            distance < 1e-3,
            "chunked decode changed distribution: {distance}"
        );
    }

    #[test]
    fn unchanged_context_adopts_new_generation_without_rebuild() {
        let current = Some("context A".to_owned());
        let mut gen = 1;
        // Context B was shed: the worker still has A when the adapter
        // switches back to A in generation 3.
        assert!(same_context_adopt_generation(
            &current,
            &mut gen,
            &Some("context A".to_owned()),
            3
        ));
        assert_eq!(gen, 3);
        assert!(!same_context_adopt_generation(
            &current,
            &mut gen,
            &Some("context B".to_owned()),
            4
        ));
        assert_eq!(
            gen, 3,
            "changed text must stamp pending replies with the old generation"
        );
    }

    #[test]
    fn unchanged_empty_context_adopts_new_generation() {
        let mut gen = 0;
        assert!(same_context_adopt_generation(&None, &mut gen, &None, 2));
        assert_eq!(gen, 2);
    }
}

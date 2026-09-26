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
use crate::LlamaConfig;

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
/// Token budget for the context block body (header and tail excluded):
/// tail-trimmed so the most recent lines always survive. Preamble (~30) +
/// block (~1050) + window (~250) stays well under `n_ctx`.
const CONTEXT_MAX_TOKENS: usize = 1024;

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
        let mut toks = self
            .model
            .str_to_token(PREAMBLE, AddBos::Never)
            .map_err(|e| format!("preamble tokenize: {e}"))?;
        if let Some(ctx) = &self.context_text {
            let head = self
                .model
                .str_to_token(CONTEXT_HEADER, AddBos::Never)
                .map_err(|e| format!("context header tokenize: {e}"))?;
            let mut body = self
                .model
                .str_to_token(ctx, AddBos::Never)
                .map_err(|e| format!("context tokenize: {e}"))?;
            let tail = self
                .model
                .str_to_token(CONTEXT_TAIL, AddBos::Never)
                .map_err(|e| format!("context tail tokenize: {e}"))?;
            let budget = CONTEXT_MAX_TOKENS.saturating_sub(head.len() + tail.len());
            if body.len() > budget {
                body.drain(..body.len() - budget);
            }
            toks.extend_from_slice(&head);
            toks.extend_from_slice(&body);
            toks.extend_from_slice(&tail);
        }
        Ok(toks)
    }

    /// Decode the pinned prefix into sequence 0, replacing the pinned
    /// empty-key branch's distribution. Callers changing the context block
    /// must flush all other branches right after: their KV shares prefix
    /// cells with seq 0, which this rewrites.
    fn rebuild_prefix(&mut self) -> Result<(), String> {
        let toks = self.prefix_tokens()?;
        self.ctx
            .clear_kv_cache_seq(Some(0), None, None)
            .map_err(|e| format!("kv clear: {e}"))?;
        let mut batch = LlamaBatch::new(toks.len(), 1);
        for (i, &t) in toks.iter().enumerate() {
            batch
                .add(t, i as i32, &[0], i + 1 == toks.len())
                .map_err(|e| format!("batch add: {e}"))?;
        }
        self.ctx
            .decode(&mut batch)
            .map_err(|e| format!("prefix decode: {e}"))?;
        self.n_preamble = toks.len() as i32;
        let probs: Arc<[f32]> = softmax(self.ctx.get_logits_ith(batch.n_tokens() - 1)).into();
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
        let tail = &tokens[k..];
        let mut batch = LlamaBatch::new(tail.len(), 1);
        for (i, &t) in tail.iter().enumerate() {
            let pos = self.n_preamble + (k + i) as i32;
            batch
                .add(t, pos, &[seq], i + 1 == tail.len())
                .map_err(|e| format!("batch add: {e}"))?;
        }
        self.ctx
            .decode(&mut batch)
            .map_err(|e| format!("decode: {e}"))?;
        Ok(softmax(self.ctx.get_logits_ith(batch.n_tokens() - 1)).into())
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

/// Preamble priming the model toward the register it will predict: ordinary
/// prose with normal capitalization and punctuation — the full alphabet
/// includes capitals, digits, and symbols behind group boxes, so the prompt
/// must not steer mass away from them. SPEC §9.
const PREAMBLE: &str = "What follows is ordinary English text, with normal capitalization \
and punctuation, as someone types it out one character at a time.\n\n";

pub fn run(
    cfg: LlamaConfig,
    alphabet: Alphabet,
    rx: Receiver<ToWorker>,
    tx: Sender<FromWorker>,
    ready: Sender<Result<(), String>>,
) {
    // llama.cpp's backend is process-global and may only be initialized once
    // (several workers exist under `cargo test`).
    static BACKEND: std::sync::OnceLock<Result<LlamaBackend, String>> = std::sync::OnceLock::new();
    let setup = || -> Result<(&'static LlamaBackend, LlamaModel), String> {
        let backend = BACKEND
            .get_or_init(|| {
                // llama.cpp chats on stderr by default; an OSK has no use for it.
                llama_cpp_2::send_logs_to_tracing(
                    llama_cpp_2::LogOptions::default().with_logs_enabled(false),
                );
                LlamaBackend::init().map_err(|e| format!("backend: {e}"))
            })
            .as_ref()
            .map_err(Clone::clone)?;
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

    let mut w = Worker {
        model: &model,
        ctx,
        vocab,
        alphabet,
        n_preamble: 0,
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
                        // seq 0 may hold a partially rewritten prefix; the
                        // flush below still applies, so prediction degrades
                        // (PPM-heavy) rather than serving stale context.
                        eprintln!("keyb-lm-llama: context prefix rebuild: {e}");
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
    use super::same_context_adopt_generation;

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

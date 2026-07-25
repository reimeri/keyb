//! Persistent, decaying user-vocabulary model.
//!
//! Unlike the session [`PpmModel`](crate::PpmModel) — whose integer counts are
//! rewound exactly by `unlearn` on retraction and reset — the user model is a
//! *long-term* record of the words this user actually commits. It uses float
//! counts so evidence can decay by wall-clock time: a word not typed for a
//! while loses weight and, once it falls below a floor, is dropped entirely.
//!
//! The model is fed via [`LanguageModel::note_commit`]/`note_retract`, which
//! the engine emits only at genuine commit/retract sites (never on reset), so
//! personalization survives `Engine::reset`. It is blended into predictions by
//! the [`Personalized`] decorator with a weight proportional to how much
//! evidence the user model actually has for a context, so contexts the user
//! has never touched leave the base model's prediction essentially unchanged.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use keyb_core::LanguageModel;

/// Escape-smoothing constants (shared with `PpmModel`; Dasher lineage).
const ALPHA: f64 = 0.49;
const BETA: f64 = 0.77;
/// Per-symbol probability floor after blending.
const P_MIN: f64 = 1e-4;
/// A symbol whose decayed count drops below this is treated as gone.
const SYM_EPS: f32 = 1e-3;
/// A context whose total decayed evidence drops below this is pruned.
const PRUNE_TOTAL: f32 = 0.5;
/// Effective observations at which a context's blend weight reaches 0.5.
const WEIGHT_K: f64 = 4.0;

/// Default forgetting half-life: ~30 days.
pub const DEFAULT_HALF_LIFE_SECS: u64 = 30 * 24 * 60 * 60;
/// Default PPM order for the user model (matches the seeded session model).
pub const DEFAULT_ORDER: usize = 4;

struct Entry {
    /// Decayed counts per symbol, as of `last_used`.
    counts: Vec<f32>,
    total: f32,
    distinct: u32,
    /// Unix seconds at which `counts` were last brought current.
    last_used: u64,
}

impl Entry {
    fn empty(n_sym: usize, now: u64) -> Self {
        Entry {
            counts: vec![0.0; n_sym],
            total: 0.0,
            distinct: 0,
            last_used: now,
        }
    }
}

/// Wall-clock decay factor for evidence last touched at `last_used`.
fn decay_factor(last_used: u64, now: u64, half_life_secs: u64) -> f64 {
    if now <= last_used || half_life_secs == 0 {
        return 1.0;
    }
    let elapsed = (now - last_used) as f64;
    0.5f64.powf(elapsed / half_life_secs as f64)
}

/// Bring `entry` current: decay its counts to `now`, drop symbols that fall
/// below `SYM_EPS`, recompute `total`/`distinct`, and stamp `last_used = now`.
/// Free function (not a method) so callers can invoke it while holding a
/// `&mut Entry` borrowed out of the context map.
fn decay_entry(entry: &mut Entry, now: u64, half_life_secs: u64) {
    let f = decay_factor(entry.last_used, now, half_life_secs) as f32;
    entry.last_used = now;
    if f >= 1.0 {
        return;
    }
    let mut total = 0.0;
    let mut distinct = 0;
    for c in entry.counts.iter_mut() {
        *c *= f;
        if *c < SYM_EPS {
            *c = 0.0;
        } else {
            total += *c;
            distinct += 1;
        }
    }
    entry.total = total;
    entry.distinct = distinct;
}

pub struct UserModel {
    order: usize,
    n_sym: usize,
    half_life_secs: u64,
    contexts: HashMap<Box<[u8]>, Entry>,
}

impl UserModel {
    pub fn new(n_sym: usize, order: usize, half_life_secs: u64) -> Self {
        UserModel {
            order,
            n_sym,
            half_life_secs,
            contexts: HashMap::new(),
        }
    }

    /// Observe `sym` after `context` in genuinely committed text.
    pub fn learn(&mut self, context: &[u8], sym: u8, now: u64) {
        let n_sym = self.n_sym;
        let half_life = self.half_life_secs;
        for k in 0..=self.order.min(context.len()) {
            let key = &context[context.len() - k..];
            let entry = self
                .contexts
                .entry(key.into())
                .or_insert_with(|| Entry::empty(n_sym, now));
            decay_entry(entry, now, half_life);
            if entry.counts[sym as usize] < SYM_EPS {
                entry.distinct += 1;
            }
            entry.counts[sym as usize] += 1.0;
            entry.total += 1.0;
        }
    }

    /// Undo a prior `learn` (the commit was retracted).
    pub fn unlearn(&mut self, context: &[u8], sym: u8, now: u64) {
        let half_life = self.half_life_secs;
        for k in 0..=self.order.min(context.len()) {
            let key = &context[context.len() - k..];
            let Some(entry) = self.contexts.get_mut(key) else {
                continue;
            };
            decay_entry(entry, now, half_life);
            let s = sym as usize;
            if entry.counts[s] > 0.0 {
                entry.counts[s] = (entry.counts[s] - 1.0).max(0.0);
                if entry.counts[s] < SYM_EPS {
                    entry.counts[s] = 0.0;
                }
                // Recompute totals from the (decayed, edited) counts so they
                // never drift against a partially-decayed removed symbol.
                let mut total = 0.0;
                let mut distinct = 0;
                for &c in entry.counts.iter() {
                    if c >= SYM_EPS {
                        total += c;
                        distinct += 1;
                    }
                }
                entry.total = total;
                entry.distinct = distinct;
            }
        }
    }

    /// User-vocabulary belief for `context`, plus a weight in `[0, 1]` giving
    /// how much evidence backs it. The distribution bottoms out at uniform;
    /// the weight — not the distribution — is what keeps unseen contexts from
    /// disturbing the base model (weight → 0 when there is no evidence).
    pub fn predict(&self, context: &[u8], now: u64) -> (Vec<f64>, f64) {
        let n = self.n_sym;
        let mut p = vec![1.0 / n as f64; n];
        let mut deepest_total = 0.0f64;
        for k in 0..=self.order.min(context.len()) {
            let key = &context[context.len() - k..];
            let Some(c) = self.contexts.get(key) else {
                continue;
            };
            let f = decay_factor(c.last_used, now, self.half_life_secs);
            let total = c.total as f64 * f;
            if total <= 0.0 {
                continue;
            }
            let denom = total + ALPHA;
            let escape = ((ALPHA + BETA * c.distinct as f64) / denom).clamp(0.0, 1.0);
            for s in 0..n {
                let cnt = c.counts[s] as f64 * f;
                let direct = if cnt > 0.0 {
                    (cnt - BETA).max(0.0) / denom
                } else {
                    0.0
                };
                p[s] = direct + escape * p[s];
            }
            // Weight comes only from *contextual* (order ≥ 1) evidence: the
            // order-0 unigram still shapes backoff above, but on its own must
            // not pull predictions, or every context the user has never typed
            // in would drift toward their global letter frequency. Ascending
            // order means the last match is the most specific.
            if k >= 1 {
                deepest_total = total;
            }
        }
        let mut sum = 0.0;
        for v in p.iter_mut() {
            *v = v.max(P_MIN);
            sum += *v;
        }
        for v in p.iter_mut() {
            *v /= sum;
        }
        let weight = deepest_total / (deepest_total + WEIGHT_K);
        (p, weight)
    }

    /// Decay every context to `now` and drop those below the prune floor.
    /// Bounds both memory and the on-disk size; cheap to run at load/save.
    pub fn sweep(&mut self, now: u64) {
        let half_life = self.half_life_secs;
        for entry in self.contexts.values_mut() {
            decay_entry(entry, now, half_life);
        }
        self.contexts.retain(|_, e| e.total >= PRUNE_TOTAL);
    }

    #[cfg(test)]
    fn context_count(&self) -> usize {
        self.contexts.len()
    }

    // ---- Serialization (compact, versioned, dependency-free) ----

    const MAGIC: &'static [u8; 4] = b"KBUM";
    const VERSION: u8 = 1;

    fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(Self::MAGIC)?;
        w.write_all(&[Self::VERSION])?;
        w.write_all(&(self.n_sym as u32).to_le_bytes())?;
        w.write_all(&[self.order as u8])?;
        w.write_all(&self.half_life_secs.to_le_bytes())?;
        w.write_all(&(self.contexts.len() as u32).to_le_bytes())?;
        for (key, entry) in &self.contexts {
            w.write_all(&(key.len() as u16).to_le_bytes())?;
            w.write_all(key)?;
            w.write_all(&entry.last_used.to_le_bytes())?;
            let pairs: Vec<(u8, f32)> = entry
                .counts
                .iter()
                .enumerate()
                .filter(|(_, &c)| c >= SYM_EPS)
                .map(|(s, &c)| (s as u8, c))
                .collect();
            w.write_all(&(pairs.len() as u16).to_le_bytes())?;
            for (s, c) in pairs {
                w.write_all(&[s])?;
                w.write_all(&c.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Parse a model from bytes. Returns `None` on any mismatch or corruption —
    /// a format/alphabet change simply resets personalization rather than
    /// failing the app.
    fn parse(bytes: &[u8], expect_n_sym: usize) -> Option<Self> {
        let mut r = bytes;
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic).ok()?;
        if &magic != Self::MAGIC {
            return None;
        }
        let mut b1 = [0u8; 1];
        r.read_exact(&mut b1).ok()?;
        if b1[0] != Self::VERSION {
            return None;
        }
        let n_sym = read_u32(&mut r)? as usize;
        if n_sym != expect_n_sym {
            return None;
        }
        r.read_exact(&mut b1).ok()?;
        let order = b1[0] as usize;
        let half_life_secs = read_u64(&mut r)?;
        let n_contexts = read_u32(&mut r)? as usize;
        let mut contexts = HashMap::with_capacity(n_contexts);
        for _ in 0..n_contexts {
            let key_len = read_u16(&mut r)? as usize;
            if key_len > order {
                return None;
            }
            let mut key = vec![0u8; key_len];
            r.read_exact(&mut key).ok()?;
            let last_used = read_u64(&mut r)?;
            let n_pairs = read_u16(&mut r)? as usize;
            let mut counts = vec![0.0f32; n_sym];
            let mut total = 0.0;
            let mut distinct = 0;
            for _ in 0..n_pairs {
                r.read_exact(&mut b1).ok()?;
                let s = b1[0] as usize;
                let c = read_f32(&mut r)?;
                if s >= n_sym {
                    return None;
                }
                counts[s] = c;
                if c >= SYM_EPS {
                    total += c;
                    distinct += 1;
                }
            }
            contexts.insert(
                key.into_boxed_slice(),
                Entry {
                    counts,
                    total,
                    distinct,
                    last_used,
                },
            );
        }
        Some(UserModel {
            order,
            n_sym,
            half_life_secs,
            contexts,
        })
    }

    /// Load from `path`, sweeping to `now`. Any absence, mismatch, or parse
    /// error yields `None` (caller starts fresh).
    pub fn load(path: &Path, expect_n_sym: usize, now: u64) -> Option<Self> {
        let bytes = std::fs::read(path).ok()?;
        let mut m = Self::parse(&bytes, expect_n_sym)?;
        m.sweep(now);
        Some(m)
    }

    /// Serialize into memory. Cheap and pure — no syscalls — so the frame
    /// thread can produce a snapshot and hand it to [`SaveWorker`] without
    /// touching the filesystem. ~0.3 ms for a session-sized model (~17k
    /// committed characters, ~74 kB).
    ///
    /// Buffering is not incidental: `write_to` emits one call per field and two
    /// per stored symbol, so serializing straight into a `File` cost hundreds
    /// of thousands of `write` syscalls — measured at ~62 ms of the ~72 ms a
    /// save used to take, dwarfing even `sync_all`.
    pub fn to_bytes(&self) -> Vec<u8> {
        // Header + a generous per-context estimate; growth is amortized anyway.
        let mut buf = Vec::with_capacity(32 + self.contexts.len() * 48);
        self.write_to(&mut buf)
            .expect("writing to a Vec is infallible");
        buf
    }

    /// Serialize atomically (temp file + rename), creating parent dirs.
    ///
    /// Blocks on `sync_all`: ~9 ms on btrfs, and unbounded on a loaded or slow
    /// disk. That is over the 8 ms frame budget on its own, so this must not be
    /// called from the frame loop — [`Personalized`] goes through
    /// [`SaveWorker`] instead.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        write_atomically(path, &self.to_bytes())
    }
}

/// Write `bytes` to `path` atomically (temp file + fsync + rename), creating
/// parent dirs. `sync_all` dominates and scales with the filesystem, not the
/// payload; callers on a latency budget run this off-thread.
fn write_atomically(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

fn read_u16(r: &mut &[u8]) -> Option<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).ok()?;
    Some(u16::from_le_bytes(b))
}
fn read_u32(r: &mut &[u8]) -> Option<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).ok()?;
    Some(u32::from_le_bytes(b))
}
fn read_u64(r: &mut &[u8]) -> Option<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).ok()?;
    Some(u64::from_le_bytes(b))
}
fn read_f32(r: &mut &[u8]) -> Option<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).ok()?;
    Some(f32::from_le_bytes(b))
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Save at most once per this many committed characters (debounced); a final
/// save also runs on `Drop`.
const SAVE_EVERY_COMMITS: u32 = 64;
/// Default weight of the user model in the blend, before per-context scaling.
pub const DEFAULT_LAMBDA_USER: f64 = 0.15;

#[derive(Default)]
struct SlotState {
    /// Newest serialized snapshot not yet on disk.
    pending: Option<Vec<u8>>,
    shutdown: bool,
}

#[derive(Default)]
struct SaveSlot {
    state: Mutex<SlotState>,
    ready: Condvar,
}

/// Off-thread persistence for the user model.
///
/// `note_commit` runs inside `Engine::step`, on the render thread. Saving there
/// blocked it for ~72 ms on btrfs once every `SAVE_EVERY_COMMITS` characters —
/// a visible stutter every few words, nine times the 8 ms frame budget
/// (SPEC §10). Buffering serialization (see [`UserModel::to_bytes`]) removed
/// ~62 ms of that, but the remaining `sync_all` still costs ~9 ms and grows
/// with disk load, so it cannot stay on the frame loop either. The frame thread
/// now only *serializes* (pure, in-memory, ~0.3 ms) and leaves the bytes here;
/// this thread does the file I/O.
///
/// The mailbox holds exactly one snapshot and is latest-wins: a snapshot that
/// has not been written yet is simply replaced. A disk slower than the typing
/// rate therefore costs staleness, never queue growth and never back-pressure
/// onto steering.
struct SaveWorker {
    slot: Arc<SaveSlot>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl SaveWorker {
    /// Spawn the writer for `path`. `None` if the thread can't be spawned, in
    /// which case the caller saves inline rather than losing the data.
    fn spawn(path: PathBuf) -> Option<Self> {
        let slot = Arc::new(SaveSlot::default());
        let worker_slot = Arc::clone(&slot);
        let handle = std::thread::Builder::new()
            .name("keyb-user-model".into())
            .spawn(move || run_saver(&path, &worker_slot))
            .map_err(|e| eprintln!("keyb: user-vocabulary writer thread: {e}"))
            .ok()?;
        Some(SaveWorker {
            slot,
            handle: Some(handle),
        })
    }

    /// Hand over a snapshot, replacing any still-unwritten one. Never blocks on
    /// I/O — only on the (uncontended, I/O-free) mailbox lock.
    fn submit(&self, bytes: Vec<u8>) {
        let mut state = lock(&self.slot.state);
        state.pending = Some(bytes);
        drop(state);
        self.slot.ready.notify_one();
    }
}

impl Drop for SaveWorker {
    /// Drain and join, so the final snapshot is durably on disk before the
    /// process exits.
    fn drop(&mut self) {
        lock(&self.slot.state).shutdown = true;
        self.slot.ready.notify_one();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// A poisoned mailbox mutex carries no invariant worth honoring — the guarded
/// state is one `Option<Vec<u8>>` — so recover rather than propagate a panic
/// into the frame loop.
fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn run_saver(path: &Path, slot: &SaveSlot) {
    // Report a failing disk once; an OSK that reprints every 64 characters is
    // worse than one that stays quiet.
    let mut reported = false;
    loop {
        let mut state = lock(&slot.state);
        while state.pending.is_none() && !state.shutdown {
            state = slot.ready.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        // Take the snapshot before honoring shutdown, so the final save lands.
        let Some(bytes) = state.pending.take() else {
            return; // shutdown with nothing left to write
        };
        drop(state);
        if let Err(e) = write_atomically(path, &bytes) {
            if !reported {
                reported = true;
                eprintln!("keyb: could not persist user vocabulary: {e}");
            }
        }
    }
}

/// Wraps any base [`LanguageModel`] and folds a persistent [`UserModel`] into
/// its predictions. The base model's session/reset behavior is untouched:
/// `learn`/`unlearn` forward through unchanged, and only `note_commit`/
/// `note_retract` (never emitted on reset) drive the persistent layer.
pub struct Personalized {
    inner: Box<dyn LanguageModel>,
    user: UserModel,
    lambda_user: f64,
    path: PathBuf,
    /// Writer thread (SPEC §10: file I/O must stay off the frame loop). `None`
    /// only if the thread could not be spawned; saves then run inline.
    saver: Option<SaveWorker>,
    commits_since_save: u32,
    dirty: bool,
}

impl Personalized {
    /// Load an existing user model for `path` (or start fresh) and wrap `inner`.
    pub fn load_or_new(n_sym: usize, inner: Box<dyn LanguageModel>, path: PathBuf) -> Self {
        let now = now_secs();
        let user = UserModel::load(&path, n_sym, now)
            .unwrap_or_else(|| UserModel::new(n_sym, DEFAULT_ORDER, DEFAULT_HALF_LIFE_SECS));
        Personalized {
            inner,
            user,
            lambda_user: DEFAULT_LAMBDA_USER,
            saver: SaveWorker::spawn(path.clone()),
            path,
            commits_since_save: 0,
            dirty: false,
        }
    }

    /// Fold the user model into a base distribution for `context`.
    fn personalize(&self, context: &[u8], base: Vec<f64>, now: u64) -> Vec<f64> {
        let (udist, weight) = self.user.predict(context, now);
        let lambda = self.lambda_user * weight;
        if lambda <= 0.0 {
            return base;
        }
        let mut sum = 0.0;
        let mut p: Vec<f64> = base
            .iter()
            .zip(&udist)
            .map(|(&b, &u)| {
                let v = (1.0 - lambda) * b + lambda * u;
                sum += v;
                v
            })
            .collect();
        if sum > 0.0 {
            for v in p.iter_mut() {
                *v /= sum;
            }
        }
        p
    }

    /// Sweep, snapshot, and queue the snapshot for the writer thread. Every
    /// step here is in-memory: sweeping ~20k contexts costs ~0.2 ms and
    /// serializing them ~0.3 ms, both comfortably inside a frame.
    fn save_now(&mut self) {
        // Sweep before serializing so stale contexts leave the file.
        self.user.sweep(now_secs());
        let bytes = self.user.to_bytes();
        match &self.saver {
            Some(saver) => saver.submit(bytes),
            // No writer thread: better a rare stall than lost personalization.
            None => {
                if let Err(e) = write_atomically(&self.path, &bytes) {
                    eprintln!("keyb: could not persist user vocabulary: {e}");
                }
            }
        }
        self.commits_since_save = 0;
        self.dirty = false;
    }
}

impl LanguageModel for Personalized {
    fn predict(&mut self, context: &[u8]) -> Vec<f64> {
        let base = self.inner.predict(context);
        self.personalize(context, base, now_secs())
    }

    fn learn(&mut self, context: &[u8], sym: u8) {
        self.inner.learn(context, sym);
    }

    fn unlearn(&mut self, context: &[u8], sym: u8) {
        self.inner.unlearn(context, sym);
    }

    fn note_commit(&mut self, context: &[u8], sym: u8) {
        self.inner.note_commit(context, sym);
        self.user.learn(context, sym, now_secs());
        self.dirty = true;
        self.commits_since_save += 1;
        if self.commits_since_save >= SAVE_EVERY_COMMITS {
            self.save_now();
        }
    }

    fn note_retract(&mut self, context: &[u8], sym: u8) {
        self.inner.note_retract(context, sym);
        self.user.unlearn(context, sym, now_secs());
        self.dirty = true;
    }

    fn poll_upgrades(&mut self) -> Vec<(Vec<u8>, Vec<f64>)> {
        let now = now_secs();
        self.inner
            .poll_upgrades()
            .into_iter()
            .map(|(ctx, dist)| {
                let p = self.personalize(&ctx, dist, now);
                (ctx, p)
            })
            .collect()
    }

    fn frame_hint(&mut self, center_context: &[u8]) {
        self.inner.frame_hint(center_context);
    }

    fn release(&mut self, context: &[u8]) {
        self.inner.release(context);
    }

    fn alphabet_len(&self) -> usize {
        self.inner.alphabet_len()
    }
}

impl Drop for Personalized {
    /// Queue a final snapshot. Dropping `saver` afterwards (fields drop after
    /// this body) drains the mailbox and joins the writer, so the last
    /// characters of a session are durable before the process exits.
    fn drop(&mut self) {
        if self.dirty {
            self.save_now();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 30;
    const HL: u64 = DEFAULT_HALF_LIFE_SECS;

    #[test]
    fn learn_boosts_symbol() {
        let mut m = UserModel::new(N, 4, HL);
        let base = m.predict(b"q", 0).0[5];
        for _ in 0..5 {
            m.learn(b"q", 5, 0);
        }
        let (p, w) = m.predict(b"q", 0);
        assert!(p[5] > base, "learned symbol should gain mass");
        assert!(w > 0.0, "evidence should produce a positive weight");
    }

    #[test]
    fn unseen_context_has_zero_weight() {
        let mut m = UserModel::new(N, 4, HL);
        m.learn(b"q", 5, 0);
        // A context the model has never seen (no prefix match either).
        let (_p, w) = m.predict(b"zzz", 0);
        assert_eq!(w, 0.0, "no evidence anywhere → no influence");
    }

    #[test]
    fn decay_halves_after_one_half_life() {
        let mut m = UserModel::new(N, 4, HL);
        m.learn(b"q", 5, 0);
        let (_p0, w0) = m.predict(b"q", 0);
        let (_p1, w1) = m.predict(b"q", HL); // one half-life later
        // Weight = e/(e+K); with e halved from 1.0 to 0.5 it must drop.
        assert!(w1 < w0, "evidence weight must fall after a half-life");
        // Effective evidence at t=HL is ~0.5, so weight ~ 0.5/4.5.
        assert!((w1 - (0.5 / (0.5 + WEIGHT_K))).abs() < 1e-6);
    }

    #[test]
    fn stale_single_use_context_is_pruned() {
        let mut m = UserModel::new(N, 4, HL);
        m.learn(b"q", 5, 0);
        assert!(m.context_count() > 0);
        // Four half-lives: 1.0 → ~0.0625, below PRUNE_TOTAL (0.5).
        m.sweep(4 * HL);
        assert_eq!(m.context_count(), 0, "long-unused evidence is dropped");
    }

    #[test]
    fn unlearn_removes_evidence() {
        let mut m = UserModel::new(N, 4, HL);
        m.learn(b"q", 5, 0);
        m.unlearn(b"q", 5, 0);
        let (_p, w) = m.predict(b"q", 0);
        assert_eq!(w, 0.0, "retracted commit leaves no evidence");
    }

    #[test]
    fn round_trip_preserves_predictions() {
        let mut m = UserModel::new(N, 4, HL);
        for _ in 0..3 {
            m.learn(b"ab", 7, 100);
            m.learn(b"", 2, 100);
        }
        let mut buf = Vec::new();
        m.write_to(&mut buf).unwrap();
        let back = UserModel::parse(&buf, N).expect("round trip");
        for ctx in [&b""[..], &b"ab"[..], &b"b"[..]] {
            let a = m.predict(ctx, 100);
            let b = back.predict(ctx, 100);
            assert_eq!(a.0, b.0, "distributions must match after round trip");
            assert!((a.1 - b.1).abs() < 1e-9);
        }
    }

    /// Unique scratch path per test; no dev-dependency needed.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "keyb-user-{}-{}-{name}",
            std::process::id(),
            now_secs()
        ));
        dir.join("user_model.v1.bin")
    }

    fn personalized(path: &Path) -> Personalized {
        Personalized::load_or_new(N, Box::new(crate::PpmModel::new(N, 4)), path.to_path_buf())
    }

    /// `note_commit` runs on the render thread inside `Engine::step`, so the
    /// periodic save must not block on it. Before this was fixed the
    /// save-boundary commit measured ~72 ms on btrfs; the budget it has to fit
    /// inside is one 8 ms frame (SPEC §10).
    ///
    /// Note the bound only bites on a filesystem where a synchronous save would
    /// actually be slow. `temp_dir()` is tmpfs on some machines, where even the
    /// old blocking path fit in a frame and nothing here would fail — this
    /// guards the budget, it does not prove the write is asynchronous.
    #[test]
    fn periodic_save_never_blocks_the_commit_path() {
        let path = scratch("latency");
        let mut p = personalized(&path);
        let mut ctx: Vec<u8> = Vec::new();
        let mut worst = 0.0f64;
        // Several save boundaries, so the timed window definitely contains one.
        for i in 0..(SAVE_EVERY_COMMITS * 4) {
            let sym = (i % N as u32) as u8;
            let t = std::time::Instant::now();
            p.note_commit(&ctx, sym);
            worst = worst.max(t.elapsed().as_secs_f64() * 1e3);
            ctx.push(sym);
        }
        drop(p);
        assert!(
            worst < 8.0,
            "slowest note_commit was {worst:.2} ms; must stay inside a frame"
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// Moving the write off-thread must not lose it: dropping drains the
    /// mailbox and joins, so the file is complete by the time drop returns.
    #[test]
    fn drop_flushes_the_final_snapshot() {
        let path = scratch("flush");
        let mut p = personalized(&path);
        for _ in 0..3 {
            p.note_commit(b"ab", 7);
        }
        assert!(!path.exists(), "no save is due before the debounce or drop");
        drop(p);

        let back = UserModel::load(&path, N, now_secs()).expect("final save must be on disk");
        let (_, w) = back.predict(b"ab", now_secs());
        assert!(w > 0.0, "the last commits must have survived the process");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// The mailbox is latest-wins, so a burst of saves converges on the newest
    /// state rather than replaying every intermediate snapshot.
    #[test]
    fn rapid_saves_persist_the_newest_state() {
        let path = scratch("latest");
        let mut p = personalized(&path);
        let mut ctx: Vec<u8> = Vec::new();
        for i in 0..(SAVE_EVERY_COMMITS * 5) {
            let sym = (i % 3) as u8;
            p.note_commit(&ctx, sym);
            ctx.push(sym);
        }
        // One more commit after the last boundary, so only the drop-time
        // snapshot contains it.
        p.note_commit(b"zz", 11);
        drop(p);

        let back = UserModel::load(&path, N, now_secs()).expect("model on disk");
        let (dist, w) = back.predict(b"zz", now_secs());
        assert!(w > 0.0, "the newest snapshot must be the one that landed");
        assert!(dist[11] > 1.0 / N as f64);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn defensive_load_rejects_bad_input() {
        let mut m = UserModel::new(N, 4, HL);
        m.learn(b"q", 5, 0);
        let mut buf = Vec::new();
        m.write_to(&mut buf).unwrap();
        // Wrong alphabet size → rejected.
        assert!(UserModel::parse(&buf, N + 1).is_none());
        // Garbage → rejected, not a panic.
        assert!(UserModel::parse(b"not a model", N).is_none());
        assert!(UserModel::parse(&[], N).is_none());
        // Wrong magic → rejected.
        let mut bad = buf.clone();
        bad[0] = b'X';
        assert!(UserModel::parse(&bad, N).is_none());
    }
}

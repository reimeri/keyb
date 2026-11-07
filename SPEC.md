# keyb — Implementation Specification

A mouse-driven, zooming on-screen text entry system in the lineage of Dasher, with a
local LLM as the probability model. Writing is continuous steering through a
landscape of nested boxes whose sizes are the model's predictive distribution;
no clicking, no discrete keys.

Status: **M1–M3 implemented, M4 in progress** (§12) · Platform: Linux desktop
(Wayland + X11); Windows/macOS not started.

This document describes the system as built. Sections that describe work not yet
implemented say so explicitly — nothing here is aspirational unless it is
labelled **Not implemented** or **Planned**. Section numbers are referenced from
code comments and must stay stable.

---

## 1. Goals

- **Fast text entry with a pointer only.** Target ≥ 25 WPM for a practiced user with
  an ordinary mouse; graceful degradation to trackball/trackpad/eye-tracker later.
  The oracle harness (§11) is what measures this: it currently gates ≥ 15 WPM on
  PPM and ≥ 25 WPM on the LLM.
- **LLM-driven prediction.** A small local model supplies the next-token
  distribution; long continuations surface naturally when probability × zoom makes
  them visually large. No cloud dependency, no network requirement at runtime
  (the default model is fetched once, on first launch — §9).
- **Universal expressiveness.** Every possible string is reachable — smoothing
  guarantees no character ever has zero probability (novel names, codes, etc.).
- **System-wide output.** Committed text is injected into whatever application has
  focus, like a real on-screen keyboard — not just an internal text box.
- **Self-revealing UI.** Zero memorization required to start; the same steering
  skill deepens with practice with no ceiling imposed by clicking.

## 2. Non-goals (v1)

- Mobile/touch, eye-tracker, and single-switch input modes (the architecture must
  not preclude them — they are alternative sources of the same 2D steering signal).
- Cloud inference, multi-language UI (the alphabet and LM are pluggable in code;
  only English ships).
- IME-grade integration with every toolkit. v1 targets the common paths listed in §8.

Note that *user-model fine-tuning* was a non-goal but a lightweight variant now
exists: a persistent, decaying user-vocabulary layer blended on top of the base
model (§6.7). It is per-user statistics, not weight updates.

---

## 3. Tech stack

| Concern | Choice | Rationale |
|---|---|---|
| Language | **Rust** (stable, edition 2021) | Sustained 120 Hz rendering + real-time inference on worker threads with no GC pauses; first-class access to Wayland/X11 platform APIs; single static binary. |
| Windowing / input | **winit** (X11 + dev window) and **smithay-client-toolkit** (Wayland layer-shell) | winit has no layer-shell support, so the Wayland OSK frontend talks to the compositor directly and bridges the raw `wl_surface` into wgpu (§7.3). |
| 2D rendering | **vello** (wgpu-based compute rasterizer) | The UI is thousands of nested rectangles and text glyphs under continuous zoom; vello renders resolution-independent vector scenes on the GPU with crisp text at any scale. |
| Text layout | **skrifa** (charmap + advances), system font via `fc-match` with hardcoded fallback paths | Labels are single characters and short ribbon strings — left-to-right, no shaping needed. Full shaping (**parley + swash**) is deferred until richer text exists on screen. |
| LLM inference | **llama.cpp** via the `llama-cpp-2` crate | GGUF models, CPU-first with optional Vulkan/CUDA offload, mature KV-cache API with **sequence branching** (needed for tree caches, §6.5). |
| Default model | **Qwen2.5-0.5B-Instruct, Q4_K_M GGUF** (~491 MB) | Sub-50 ms per forward pass on a mid-range CPU; far below chat quality bar but far above the entropy bar needed for box sizing. Swappable via `KEYB_MODEL`; any GGUF works. |
| Fallback char model | Order-4 **PPM** trained on an embedded seed corpus + online commits, in-crate | Smoothing floor and offline/degraded mode; tiny and instant. Also the sole model when no GGUF is available. |
| Text injection | Wayland `zwp_virtual_keyboard_v1` + `zwp_input_method_v2` (one hybrid backend); X11 XTEST (via `x11rb`); in-process buffer as last resort | See §8. |
| Config/persistence | Environment variables + an editable UTF-8 system prompt under `$XDG_CONFIG_HOME/keyb/` (§9), and a binary user-vocabulary file under `$XDG_DATA_HOME/keyb/`. General-purpose TOML via `serde` is planned (M4). | |
| Async/threading | std threads + `crossbeam-channel`; **no tokio** | The concurrency shape is a small fixed set of workers with bounded channels; an async runtime adds nothing. |

Threads at runtime: the UI/frame thread, the llama.cpp inference worker
(§6.4), the injection worker (§8), and the user-model save worker (§6.7).

Why not the alternatives:

- **Tauri/Electron**: an OSK window must never take keyboard focus and needs
  layer-shell/override-redirect placement; web-view windows fight this. LLM and
  injection would live in a native sidecar anyway, leaving the web layer as pure
  overhead between the pointer and the renderer.
- **Qt/C++**: viable (original Dasher lineage), but we'd hand-roll the same
  worker/channel architecture with less safety, and QPainter/QML text under
  continuous zoom is weaker than a vector-first GPU renderer.
- **egui**: the intended toolkit for a settings window (§7.4) when one is built;
  its immediate-mode text/tessellation path is not suited to a full-screen scene
  of continuously rescaling glyphs, so it will never render the canvas.

## 4. Repository layout

Cargo workspace:

```
keyb/
├─ SPEC.md
├─ README.md
├─ start.sh                  # cargo run -p keyb-app --bin keyb
├─ Cargo.toml                # workspace
└─ crates/
   ├─ keyb-core/             # §5, §6.1: geometry, dynamics, node tree, alphabet,
   │                         #   LM trait, oracle sim. No I/O, no GPU, no deps.
   ├─ keyb-lm-ppm/           # PPM fallback model + persistent user vocabulary (§6.7).
   │  └─ assets/seed.txt     #   embedded seed corpus (include_str!)
   ├─ keyb-lm-llama/         # §6.2–6.6: llama.cpp backend, KV branch cache,
   │                         #   marginalization, speculative expansion worker.
   ├─ keyb-ui/               # §7: layer-shell + winit frontends, vello scene, input.
   ├─ keyb-inject/           # §8: platform text output backends behind one trait.
   └─ keyb-app/              # binary: wiring, model provisioning, injection worker.
```

There is no top-level `assets/` directory: the PPM seed corpus lives inside
`keyb-lm-ppm`, and fonts are discovered from the system rather than shipped.

Dependency rules as built:

- `keyb-core` has **no dependencies at all** (std only) — not even serde.
- Everything else depends on `keyb-core`.
- `keyb-ui` and `keyb-lm-*` never depend on each other; they meet only through
  channels owned by `keyb-app`. `keyb-ui` likewise never depends on
  `keyb-inject` — committed text leaves the UI through a `TextSink` callback.
- `keyb-lm-llama` does depend on `keyb-lm-ppm`, because the llama adapter uses
  PPM as its own smoothing and deadline-miss fallback (§6.3).

## 5. Core model (keyb-core)

### 5.1 Interval tree

Text space is the unit interval `[0,1)` **of the current frame** (§5.3). Each node
owns a subinterval of its parent proportional to its conditional probability;
concatenating the leaf labels on the path from the frame root to a point spells the
text. This is arithmetic coding made visible — `keyb-core` maintains the
visible/needed portion of this infinite tree:

```rust
enum NodeLabel {
    Char(u8),    // a writable leaf symbol (index into the alphabet)
    Group(u32),  // an expandable group box (§5.4)
}

struct Node {
    label: NodeLabel,
    opens: u32,              // layout node its children are drawn from
    lo: f64, hi: f64,        // interval within the *frame* (§5.3)
    depth: usize,            // below the frame root
    parent: usize,
    children: Vec<usize>,
    expanded: bool,
    alive: bool,             // false = evicted; slot reclaimed by compaction
    anim_target: Option<Box<[f64]>>,  // child partition an upgrade animates toward (§6.3)
}
```

There is no per-node `state` enum and no `lm_handle`: expansion is a bool,
eviction is a liveness flag plus a dead-slot counter, and the KV-cache branch
for a context lives in the inference worker's own map, keyed by token sequence
(§6.5). The engine never sees branch ids.

- **Expansion is lazy and screen-driven**: a node is expanded (children created,
  LM queried) when its projected screen height exceeds `expand_px` (default 40 px),
  and collapsed/evicted when below `evict_px` (default 2 px) or off-screen.
  Probability thresholds are never used directly — screen size *is*
  cumulative probability × zoom, which is the interface's native relevance measure.
- Node IDs index a slab arena. Eviction marks subtrees dead; the arena is
  compacted (live subtree copied, dead slots dropped) when it reaches
  `max_nodes` (12 000) or is more than half dead. Compaction changes no
  geometry, so it is invisible.
- Expansion is **bounded per frame and per depth**, which the original design
  did not anticipate. A confident (peaked) model would otherwise deepen the tree
  every frame until the arena filled and expansion stopped entirely — boxes
  vanish. Three limits cooperate:
  - `MAX_EXPANSIONS_PER_MAINTAIN` (64) expansions per pass;
  - a *demand* pass (already-visible, screen-large, childless boxes) that may
    exceed the speculative cap by `COMMITTED_LOOKAHEAD_LEVELS` (8) but never
    grows unboundedly, because newly created children are not re-visited within
    the same pass;
  - a *lookahead* pass that spends whatever budget remains, capped at
    `soft_advance_depth + 8` or the committed/release-ready depth + 8.
  `max_depth` (64) is the hard ceiling.

### 5.2 Steering dynamics

Standard Dasher dynamics, kept deliberately boring because they're proven:

- Pointer position is read every frame in window coordinates and mapped into the
  model frame. **Horizontal displacement from the origin line controls zoom rate**
  (right = forward, left = backward/undo); **vertical position selects** which
  interval the view center drifts toward.
- Per frame the view scales by `2^(-speed_bits · dx · dt)` toward the point
  selected by `y`, where `speed_bits` is the zoom rate in bits/second (the view
  halves per `1/speed_bits` s at full deflection). View width is clamped to
  `[1e-9, 4.0]`.
- The frontend applies a **mild expansion** around the vertical center
  (`y = 0.5 + (y_raw − 0.5) · 1.15`, clamped to `[−0.15, 1.15]`) rather than the
  compressive non-linearity of classic Dasher: the aim is the same — edge
  targets stay reachable — but the mapping over-travels at the extremes instead
  of compressing them (§7.1).
- **Vertical pan is independent of zoom** (M1 finding): the point under the
  pointer drifts toward the view center at `pan_gain · speed_bits`, so pure vertical
  pointer motion scrolls without zooming. Without this term, vertical position
  only picks the zoom fixed-point and the user cannot "move up" — they must
  zoom out and back in, which testers immediately flagged.
- **Backward zoom is the primary undo.** Zooming back across a commit boundary
  retracts committed text (§8.3). Retraction is debounced symmetrically with
  commit (the view must stay outside the deepest committed node for the
  debounce period, one symbol per period) — a hair-trigger retract turns
  steering overshoot into a commit/retract limit cycle, found via the oracle
  harness in M1. Retraction touches only the text buffer and the LM
  (`unlearn`), never geometry, so backing out cannot flash or reshuffle boxes.
- **Release commit** (not in the original design; added after testing): when
  steering disengages, any crosshair-path character box covering at least
  `commit_release_frac` (0.5) of the view commits immediately, ahead of the
  containment debounce — you can steer at a target and let go rather than zoom
  all the way in, which matters for landing a space or period. Three gates keep
  this from writing text the user merely hovered over: the engagement must have
  accumulated `release_min_zoom_bits` (0.25) of forward zoom, at most
  `release_max_chars` (3) characters land per release, and boxes below the
  fraction stay uncommitted. Retraction hysteresis is widened symmetrically
  (a committed-on-release box still "holds" the view while it covers the
  release fraction) so pausing after a release commit cannot fire a spurious
  backspace.
- **Adaptive speed**: max zoom rate rises multiplicatively while steering is
  smooth and is cut immediately on direction reversals or lateral jitter — a port
  of Dasher's auto speed control, bounded by `[min_bits_per_sec, max_bits_per_sec]`.
  **Off by default**; enabled with `KEYB_ADAPTIVE` so the fixed-speed feel and
  the oracle harness are unchanged.
- **Pause/engage**: primary mouse button held = steering engaged; released =
  frozen. A pointer that leaves the OSK column also disengages (fail-safe against
  a stuck drag). Toggle mode for users who cannot hold a button is **not
  implemented**.
- **Reset**: `Engine::reset()` returns the tree, view, and prediction context to
  their start state and rewinds the LM by replaying `unlearn` in reverse commit
  order. It emits no `TextEvent`, so text already injected into the target
  application is deliberately left alone. Exposed as an on-canvas button (§7.1).

Dynamics are a pure function `step(state, pointer, dt) -> state` — no I/O, fully
deterministic, unit-testable, and replayable from logged pointer traces (§11).

Defaults (`EngineParams`): `max_bits_per_sec` 4.5, `pan_gain` 0.8, `expand_px` 40,
`evict_px` 2, `commit_debounce` 0.15 s, `rebase_view_w` 1e-6,
`soft_advance_depth` 16, `max_depth` 64, `max_nodes` 12 000,
`upgrade_anim_secs` 0.12, `retract_eps` 0.02, `commit_release_frac` 0.5,
`release_min_zoom_bits` 0.25, `release_max_chars` 3, `adaptive_speed` false,
`min_bits_per_sec` 1.5, `speed_rise_per_sec` 0.35, `speed_drop` 0.6.

### 5.3 Numerics

Deep zooms exhaust f64 mantissa in the root frame. The frame is **decoupled
from commits** (M1 finding — rebasing on every commit rebuilt the tree and
made retraction flash): the tree survives commits and retractions untouched,
and the frame moves only when numerics or the view demand it:

- **Advance (forward)**, on either of two triggers: the view spans
  < `rebase_view_w` (1e-6) of the frame (f64 precision), or the whole view is
  contained `soft_advance_depth` (16) tree levels below the frame root
  regardless of absolute width. The second trigger exists because a confident
  model drives the frontier into the `max_depth` wall otherwise, stranding the
  crosshair in a childless box ("no boxes shown"). The root descends into the
  contained child and coordinates are affinely rescaled in place (no
  re-expansion, no visible change).
  - If the view has zoomed entirely inside a leaf that the containment debounce
    has not committed yet (fast steering outran it), advancing commits that
    character on the spot so depth stays bounded. A back-zoom still retracts it
    through the ledger.
- **Climb out (backward)**: if the view escapes the frame, one ledger level
  pops immediately. The ledger stores each absorbed level's *complete child
  partition*, so the parent is re-laid out exactly as it was on screen —
  immune to LM drift — and the current tree is transplanted into its slot.
  With an empty ledger the view clamps to the document root instead.

A grouped character (§5.4) is absorbed as two ledger levels (group box, then
leaf), so `frame_chars` (committed characters absorbed into the frame) is
tracked separately from `ledger.len()`.

This caps error without arbitrary-precision arithmetic and keeps rendering
continuous through commit, undo, and rebase.

### 5.4 The alphabet is not just letters

The alphabet is two layers:

- a flat, fixed-order list of **leaf symbols** — the writable characters, and the
  only thing the language model ever sees;
- a **layout** grouping those leaves for display: frequent letters at the top
  level, rarer runs inside expandable group boxes, as in classic Dasher.

Grouping is purely structural: a leaf's cumulative interval is `P(leaf)` whether
or not it sits inside a group, so steering feel and oracle WPM on lowercase text
are unchanged by it. A group only inserts an intermediate container box on the
path to its members.

Two alphabets are built into `keyb-core`:

- `english_v1()` — 30 leaves, flat: `a–z`, space, `.`, `,`, `'`. The shape the
  engine and LM tests are written against.
- `english_full()` — 83 leaves: the v1 30 in identical order (so v1 is a strict
  prefix), then `A–Z`, `0–9`, and 17 symbols (`! ? ; : " - ( ) / @ # & * + = _ %`),
  each run behind its own group box (labelled ⇧, `[1]`, `[!]`). This is what the
  binary runs.

Each group box is guaranteed a minimum share of its parent interval
(`group_prior`, default 0.02) so a lowercase-biased model cannot collapse it to
the smoothing floor and make it unaimable. This is the "configured prior" of the
original design, applied when the layout partitions a distribution
(`Alphabet::item_weights`) rather than inside §6.3's smoothing.

Leaf *ordering is fixed forever* (alphabetical within groups); only sizes change.

**Not implemented:**

- **Action nodes** — `DeleteWord`, `NewLine`, `CapitalizeNext` as ordinary
  steerable boxes with configured priors. Nothing in the tree commits anything
  but a character today. Planned for M4.
- **Alphabet definitions as data.** Both alphabets are hardcoded in
  `keyb-core::alphabet`; there is no TOML alphabet file, so changing the symbol
  set requires a rebuild. Planned alongside the config file (§9).

## 6. Language model subsystem

### 6.1 The trait

`keyb-core` consumes only *character-level* distributions over the alphabet.
Token thinking is entirely encapsulated in `keyb-lm-llama`. Contexts are
absolute symbol sequences (`&[u8]`), not an opaque handle.

```rust
pub trait LanguageModel: Send {
    /// P(next symbol | context). Exactly `alphabet_len` entries, all > 0, ~sums to 1.
    /// May answer with a fallback distribution and upgrade later (§6.3).
    fn predict(&mut self, context: &[u8]) -> Vec<f64>;

    /// `sym` followed `context` in committed text (reversible bookkeeping).
    fn learn(&mut self, context: &[u8], sym: u8) {}
    /// Exact inverse of `learn`, called in reverse commit order.
    fn unlearn(&mut self, context: &[u8], sym: u8) {}

    /// Genuine user commit/retract, paired with a `TextEvent`. Unlike
    /// `learn`/`unlearn` these are *not* emitted by `Engine::reset`, so
    /// persistent personalization (§6.7) can key off them and survive resets.
    fn note_commit(&mut self, context: &[u8], sym: u8) {}
    fn note_retract(&mut self, context: &[u8], sym: u8) {}

    /// Distributions that arrived asynchronously since the last call.
    fn poll_upgrades(&mut self) -> Vec<(Vec<u8>, Vec<f64>)> { Vec::new() }

    /// Once per step: context of the deepest expanded node under the view
    /// center. Async backends re-prioritize and speculate from it (§6.6).
    fn frame_hint(&mut self, center_context: &[u8]) {}

    /// Auxiliary reply-to block captured from the selection (§6.8). `None`
    /// restores committed-text-only prediction. Default: ignore.
    fn set_context(&mut self, context: Option<&str>) {}

    /// The node for `context` was evicted; cancel pending work (§6.4).
    fn release(&mut self, context: &[u8]) {}

    fn alphabet_len(&self) -> usize;
}
```

The interface is **synchronous** — PPM answers in microseconds, and the LLM
adapter satisfies it by serving a cached or fallback distribution immediately and
delivering the real one later through `poll_upgrades`. The engine never blocks.

### 6.2 Token → character marginalization

The LLM predicts subword tokens; the UI must stay character-honest (a user may
steer through the middle of what the model tokenized as one unit). Per request:

1. Decode the windowed context (§6.5) to text and tokenize it.
2. **Token healing**: roll the token sequence back to the earliest boundary
   whose remaining bytes can still be extended by some vocabulary token
   (`VocabIndex::healing_suffix`). The rolled-back tail `s` is the intra-token
   prefix the user has spelled into.
3. Take the branch distribution at that boundary (softmax over the full vocab,
   §6.5) and sum it into per-character mass: `p(next char c) = Σ p(token t)` over
   tokens whose bytes start with `s + c`. The vocabulary is stored as a sorted
   byte-string index, so this is a binary-search range per prefix rather than a
   scan. Only characters in the alphabet contribute; out-of-alphabet mass is
   dropped and the result renormalized.
4. Tokens that end *exactly* at `s` carry no next character. When that exact mass
   exceeds `EXACT_MASS_THRESHOLD` (5 %) of the total, the worker pays one extra
   forward pass per such token — descending into its actual branch — and folds
   the result in. Below the threshold it is ignored as noise.

Descending *within* a token costs **zero additional forward passes**: the parent
branch's vocabulary distribution answers every intra-token prefix, and every
computed context distribution is memoized in the worker (`computed`, capped at
16 384 entries, halved rather than cleared when full). Unlike the original
design there is no char-trie object cached on the tree node — the engine
requests one distribution per character position and the worker's caches make
that cheap.

### 6.3 Smoothing

```
p_final(c) = (1 − λ) · p_llm(c) + λ · p_ppm(c),   then floor at p_min and renormalize
```

- `λ = 0.05` default; `p_min = 1e-4` per alphabet symbol.
- Blending happens at *serve* time, not when the LLM answer is cached, so PPM's
  online learning is never served stale.
- An all-zero LLM answer (everything out-of-alphabet) degrades to PPM alone.
- Group-box priors are applied by the layout, not here (§5.4).
- If the LLM worker misses its deadline (§6.6) — which by definition it does on
  any cache miss — the frame uses `p_ppm` alone and upgrades in place when the
  real distribution arrives: sizes animate exponentially to the new partition
  over `upgrade_anim_secs` (~120 ms) so boxes never pop. A single upgraded
  character context re-lays out both the character-position node *and* any
  expanded group boxes beneath it, since they partition the same distribution.
- The persistent user-vocabulary layer (§6.7) wraps whatever model is underneath
  and blends on top of this.

### 6.4 Inference worker

One dedicated thread owns the llama.cpp context. Protocol over bounded (512)
channels, keyed by **character context**, not node id:

```
UI → LM : Predict { ctx }              // a node was expanded with fallback sizing
UI → LM : Hint    { ctx }              // deepest expanded node under the view center
UI → LM : Cancel  { ctx }              // node evicted; drop pending work
UI → LM : SetContext { text, gen }     // replace/clear the pinned context block (§6.8)
LM → UI : FromWorker { ctx, gen, dist: Option<Vec<f64>> }
```

- `dist: None` means "not serviced" (compute failed, or the queue shed it). Every
  `Predict` gets **exactly one** reply, so a context can never strand on fallback
  sizing forever. `gen` is the worker's context-block generation at reply time
  (§6.8); the adapter drops replies whose generation is not current.
- Sends from the UI use `try_send` and are dropped under back-pressure rather
  than blocking the frame loop; the adapter only records a request as in-flight
  when it actually entered the queue, so a shed request is re-requestable.
- Priority is **recency-and-locality**, not projected screen area: the worker
  picks the newest pending request that lies on (or is a prefix of) the hinted
  path, else the newest overall. Control traffic is drained before every unit of
  work, so cancellation and re-prioritization act between forward passes.
- The pending list is capped at `MAX_PENDING` (256); overflow sheds the *stalest*
  request with an explicit unserviced reply.

Screen-area-weighted priority remains the design intent but is not implemented —
the UI does not send areas, and the hint plus newest-first ordering has been
sufficient in practice.

### 6.5 KV-cache branching

llama.cpp sequence IDs implement the tree cache, over a unified KV cache of
`n_ctx` = 4096 cells with `n_seq_max = max_branches + 8`:

- Branches are keyed by **token sequence**. Expanding a context finds the longest
  cached prefix, does `copy_kv_cache_seq` from it (copy-on-write prefix reuse),
  and decodes only the divergent tail. Each branch stores its full softmaxed
  next-token distribution, which is what §6.2 marginalizes.
- Budget: `max_branches = 32` sequences beyond the pinned preamble. Eviction is
  **plain LRU** (not area-weighted — the worker has no screen geometry), never
  evicting the pinned branch or the copy source of the request in flight.
  A failed decode returns its sequence id to the free list, so the branch budget
  cannot leak.
- Sequence 0 is pinned and holds the **system preamble** — the full static
  prompt loaded from the user's instructions file (§9), followed by any
  auxiliary context block (§6.8). Its default primes the model toward ordinary
  prose with normal capitalization and punctuation. No hidden instructions are
  prepended to the user's prompt. It is warmed at startup so the first
  prediction is one prefix copy plus a short decode away. Prefix and branch
  decoding are chunked to llama.cpp's logical batch limit, with absolute token
  positions and logits requested only for the final token. Committed text is
  *not* separately resident; a re-entered branch is recomputed from the pinned
  prefix plus its token prefix.
- Context window: the worker feeds the model the last `WINDOW` (800) committed
  symbols, with the window start moving in `WINDOW_STEP` (256) jumps so branch
  prefixes stay shareable across jumps.
- The design's "on commit + rebase, free all branches not under the committed
  node" is **not implemented**: the worker is never told about commits, and LRU
  plus the branch budget reclaim the same space without that coupling.

### 6.6 Speculative expansion

The user steers at seconds-per-word; the model runs at tens of ms per token —
spend the surplus so zooming never stalls:

- When no real request is pending, the worker speculates from the hinted context:
  the **top-k (k = 4) next symbols by probability**, then deepens along the argmax
  chain up to `SPECULATE_DEPTH` (6) levels. Ranking is by model probability, not
  screen area, since the worker sees no geometry.
- Exactly **one forward pass per loop iteration**, after which control traffic is
  re-drained — a real request always preempts speculation within one pass. There
  is no batched multi-child decode.
- Speculative answers are sent to the adapter, which caches them silently; only
  contexts that were actually served with fallback sizing trigger a re-partition.
- Deadline discipline: the worker never blocks the UI; the UI never waits. A node
  whose distribution hasn't arrived renders with PPM sizing (§6.3).
- Perf budget: forward pass ≤ 50 ms p95 on the configured threads
  (`available_parallelism()/2`, clamped to 2–8); full pipeline
  (request → applied on node) ≤ 80 ms p95. Only the latter is asserted in tests
  (§11).

### 6.7 Persistent user vocabulary

A decorator (`Personalized`) wraps whatever base model is loaded — LLM or PPM —
with a long-term record of what this user actually writes. Enabled by default;
`KEYB_NO_PERSONALIZE` opts out.

- Fed exclusively by `note_commit`/`note_retract`, so it survives `Engine::reset`
  and ignores the reversible `learn` bookkeeping.
- Order-4 PPM structure with **float counts that decay by wall-clock time**
  (half-life ~30 days). Symbols below `SYM_EPS` and contexts below `PRUNE_TOTAL`
  are dropped, so the store stays bounded and stale vocabulary fades.
- Blended with weight `DEFAULT_LAMBDA_USER` (0.15) **scaled per context** by how
  much evidence exists there (half weight at `WEIGHT_K` = 4 effective
  observations), so contexts the user has never touched leave the base
  prediction essentially unchanged.
- Persisted to `$XDG_DATA_HOME/keyb/user_model.v1.bin`, debounced to at most one
  save per 64 committed characters plus a final save on drop. **Serialization
  happens on the frame thread (~0.3 ms); the file write happens on a dedicated
  save thread** with a latest-wins single-slot mailbox — an earlier inline
  implementation blocked the frame loop ~72 ms on btrfs, nine times the frame
  budget (§10). A slow disk therefore costs staleness, never back-pressure.

### 6.8 Auxiliary context block

The user can pin a block of text they are *replying to* into the LLM prefix,
between the preamble and the committed (injected) text. The block is not part of
the committed buffer and is never typed into the target application.

Capture is explicit: the **⧉** ribbon control reads the primary selection,
falling back to the regular clipboard, and hands the result to
`Engine::set_context`. Empty captures are rejected in the UI with a flash
message. The UI keeps the last 16 KiB (`CONTEXT_MAX_BYTES`) on a char boundary
so recent lines always survive; the worker applies a tighter token budget on
top.

`LanguageModel::set_context(Option<&str>)` replaces the block; `None` restores
preamble-only conditioning. Models without support ignore it (the trait default).
`Personalized` forwards to its inner model. `Engine::set_context` forwards and
then re-requests distributions for currently visible expanded nodes (largest
first, deduped, capped at 128) so boxes re-partition through the normal upgrade
path. `Engine::reset` clears the block as well as zoom state; the ctx-chip
right-click clears only the block.

The llama adapter treats the block as a **generation**: each accepted change
bumps a counter carried in `ToWorker::SetContext`, clears cache and in-flight
markers, and drops worker replies stamped with any other generation. A shed
`SetContext` is retried from a latest-wins outbox on `poll_upgrades`. Blank or
unchanged text is a no-op.

The worker retains the configured static prompt tokens across context changes
and tokenizes header + body + tail separately:

- header: `The writer is replying to the following text:\n`
- tail: `\n\nThe writer's reply:\n`
- body token budget 1024 (header and tail subtracted), **tail-trimmed**

It answers every queued `Predict` as unserviced under the old generation,
releases all old KV sequences before rebuilding sequence 0, and invalidates
user-text branches plus computed distributions. This avoids retaining the old
prefix while allocating the new one. If rebuilding fails, the worker exits
rather than serving stale logits, and the adapter falls back to PPM.

UI: a `ctx N` chip appears while a block is active. A preview panel shows the
post-cap text for ~6 s after capture (hidden while steering); left-clicking the
chip pins or dismisses it, right-clicking drops the block. Failures flash for
~5 s. For verification the exact bytes promised to the model are also dumped to
`$XDG_STATE_HOME/keyb/last_context.txt`.

## 7. Rendering (keyb-ui)

### 7.1 Scene

- Rebuild the vello scene every frame from the visible tree (immediate-mode over
  retained tree). Visible node count is bounded by the expand/evict thresholds,
  so scene building is trivially within budget.
- Boxes: nested rounded rects spanning `[x0, width]`, where `x0` tracks box
  height. Left edges are widened bottom-up so a box's label strip is never
  covered by its nearest child — without this, a high-probability chain
  (`t→h→e`, each ~99 % of its parent) hides every parent's glyph.
- **Colors are a placeholder, not the classic Dasher scheme**: hue is a
  golden-angle function of the leaf symbol (so vertical neighbours differ
  clearly) and depth alternates lightness only. A colorblind-safe designed
  palette is still to do. Group boxes get distinct, highly saturated hues
  (capitals cyan, digits amber, symbols magenta) and render as *containers* —
  faint tint, bright category-colored border — so they never read as a writable
  letter. The box under the view center gets a bright outline.
- **Text LOD**: labels appear when box height ≥ 18 px, at font size
  `0.55 × height` clamped to 13–64 px (so glyph scaling saturates around 116 px
  of box height). Below the threshold boxes are color-only. Appearance is a hard
  cut, not a fade.
- **Committed text ribbon along the top edge** (56 px), showing the tail that
  fits, plus a caret. The design called for a left-edge ribbon and a separate
  full-sentence echo strip; the built UI has one top ribbon and **no echo strip**.
- **Pending preview** (M1 finding): the ribbon also shows, dimmed past the
  caret, the uncommitted text under the crosshair path, and the boxes on that
  path get a bright edge. Letters appear the instant they are steered over;
  the commit debounce then merely brightens them. Without this, users can't
  tell when input actually happens.
- Steering reference: a vertical origin line at 18 % of the canvas width, a
  crosshair tick at the vertical center, and — while engaged — a line from the
  crosshair to the cursor.
- Five on-canvas controls in the ribbon: **↺** resets the engine (§5.2), **›**
  collapses the OSK to an edge handle (§7.3), **⎘** copies all finalized
  committed text (excluding the dim pending preview) to the system clipboard,
  and **⧉** captures the primary selection (clipboard as fallback) as the LLM's
  auxiliary context block (§6.8) at top-left; **×** at top-right quits keyb
  completely. Native Wayland copy uses `ext-data-control`/`wlr-data-control`,
  falling back to X11 clipboard access when available and logging an error
  otherwise. A speed readout sits bottom-left.

### 7.2 Frame loop

Per frame: read pointer → `keyb_core::step()` (which internally drains LM
upgrades, animates sizes, and emits expand/evict/hint traffic) → drain text
events into the sink → build scene → render. **Frame budget 8 ms** (comfortable
at 120 Hz, safe at 60 Hz). Pacing is vsync: `winit` redraw-on-vsync for the
winit frontend, Wayland frame callbacks for the layer-shell frontend.

Transient surface failures are classified explicitly: `Timeout` retries,
`Outdated`/`Lost` reconfigure and retry, `OutOfMemory` is fatal.

### 7.3 Window behavior

- Wayland: **wlr-layer-shell** surface (overlay layer, full-height column
  anchored to the **right edge**, ~1/3 of the output width, **no exclusive
  zone** so it floats over targets without shrinking them) — the Dasher canvas
  wants height, not width. `keyboard_interactivity: none` so the window can
  *never* steal keyboard focus. winit has no layer-shell, so this frontend
  speaks smithay-client-toolkit directly and bridges the raw `wl_surface` into
  wgpu. `KEYB_SHELL` overrides the frontend; `KEYB_WIDTH` (logical px) overrides
  the column width (default: 1/3 of the output). On Niri, when `NIRI_SOCKET` is
  available, an IPC event stream resolves the keyboard-focused window through
  its workspace to an output; focus on another output recreates the layer
  surface there while preserving engine and text state. IPC loss retains the
  last output and reconnects in the background. Other Wayland compositors
  retain the startup output.
  If the layer shell fails at startup, the frontend falls back to a plain
  window with a message.
- X11: winit window with override-redirect + `_NET_WM_WINDOW_TYPE_DOCK`
  (unmanaged ⇒ the WM never focuses it), same right-edge full-height column.
- `KEYB_SHELL=window` gives an ordinary resizable window — the convenient mode
  for development.
- **Dismissal**: the OSK can be collapsed to a small edge handle (28 × 110 px),
  which freezes the engine and draws only the handle; clicking the handle
  restores it. The separate **×** control exits the process, destroying the
  native surface and all rendering resources; reopening requires launching
  keyb again.
- **Pointer capture is not implemented.** The design calls for pointer-lock while
  steering so the cursor can't wander off the OSK mid-word; both frontends
  instead disengage steering when the pointer leaves the column (fail-safe
  against a stuck drag). Still open, deferred from M3 to M4.

### 7.4 Settings

**Not implemented.** There is no settings window; `egui` is not a dependency.
Runtime configuration today is environment variables and the startup-loaded
instructions file (§9), plus on-canvas affordances: the scroll wheel adjusts the
speed cap (clamped to 1–10 bits/s) in both frontends, and `+`/`-` and `Esc`
additionally work in the winit frontend
(the layer-shell surface takes no keyboard input by design).
A separate `egui` window covering model selection,
speed caps, alphabet, injection backend, and telemetry opt-in remains the plan
for M4; it must never be visible during typing.

## 8. Text output (keyb-inject)

```rust
pub trait Injector: Send {
    fn commit(&mut self, text: &str) -> Result<()>;
    /// Remove exactly `text` from the end of what was committed. (M3: the
    /// text rather than a count — IM deletion needs byte lengths, key
    /// synthesis needs char counts.)
    fn retract(&mut self, text: &str) -> Result<()>;
    fn capabilities(&self) -> Caps;                     // true retraction?
}
```

Backends, auto-detected in order (`KEYB_INJECT` forces one):

1. **Wayland `zwp_input_method_v2`** (preferred where the compositor allows it):
   real commit + `delete_surrounding_text` → true retraction.
2. **Wayland `zwp_virtual_keyboard_v1`**: synthesizes key events from a
   generated keymap in which every alphabet symbol owns a keycode
   (layout-independent); retraction = synthesized backspaces.
3. **X11 XTEST**: synthesized key events, keycodes resolved from the user's
   live layout at runtime (verified against a Colemak host); retraction =
   backspaces. Characters the layout cannot produce are reported as errors.
4. **Buffer fallback**: an in-process buffer that accumulates text and echoes it
   to stderr — it injects nothing, and is also the deterministic test double.
   The originally planned **clipboard-assist (type-via-paste) backend does not
   exist**; the buffer took its place as the last resort.

M3 findings (from the foot-under-headless-sway harness, §11):

- Backends 1 and 2 are **one hybrid injector on one connection**, choosing
  *per operation*: IM activation tracks the focused window, so "which backend"
  is not a startup decision. Commits ride the IM whenever the focused field
  activated us, key synthesis otherwise.
- `delete_surrounding_text` is only trustworthy when the target has sent
  `surrounding_text` (terminals never do — they silently drop deletions), so
  retraction falls back to BackSpace synthesis even while the IM is active
  unless surrounding text was seen. Accordingly `Caps::true_retraction` is
  reported true only when no virtual keyboard is in play *and* the IM is usable
  *and* surrounding text has been seen.
- Targets need ~60 ms after activating the IM before their first commit
  reliably lands (foot drops a commit that races its IME setup); the injector
  enforces that settle delay. Irrelevant during interactive steering.

### 8.3 Commit semantics

- Text is committed when its node's interval fully contains the view frame
  (i.e., the user has zoomed entirely inside it) **and** has stayed contained
  for 150 ms (debounce against overshoot) — or on release, under the gates in
  §5.2, or when a frame advance absorbs an uncommitted leaf (§5.3).
- A rolling **retraction ledger** of the last 32 commits is kept (in the
  injection worker, keyb-app side); backward zoom across a commit boundary
  pops the ledger and calls `retract()`. The ledger verifies the whole suffix
  before mutating, and retraction beyond what it tracks is refused — the engine
  still retracts internally, but stale text in the target is left standing
  rather than guessed at. Injectors without true retraction use backspace
  synthesis; when capabilities report unreliable retraction the engine's
  hysteresis is widened (`retract_eps` 0.02 → 0.10) so retracting takes a firmer
  back-zoom.
- The injection worker runs on its own thread behind an unbounded channel, so a
  slow compositor roundtrip can never stall the frame loop.

## 9. Configuration & persistence

**There is no general-purpose config file yet.**
`$XDG_CONFIG_HOME/keyb/config.toml` (model path, threads, speed, alphabet,
injection preferences, novice mode) is planned for M4 along with the settings
UI (§7.4) and data-driven alphabets (§5.4). The static LLM prompt has its own
editable text file; other settings are environment variables or compiled-in
defaults.

### Editable static prompt

- `$XDG_CONFIG_HOME/keyb/instructions.txt`, falling back to
  `$HOME/.config/keyb/instructions.txt`. Empty or relative XDG paths are ignored;
  HOME must also be absolute. An unavailable config directory is nonfatal.
- Loaded once at startup by `keyb-app`, which creates the parent directory and
  default file when missing. Creation must not overwrite an existing file,
  including a dangling symlink. Deleting the file and restarting restores the
  default. Edits require an app restart; there is no live reload or editor UI.
- The file contains the **entire static prompt**, initially:
  `What follows is ordinary English text, with normal capitalization and punctuation, as someone types it out one character at a time.`
  The initial file ends with two newlines, matching the previous compiled-in
  preamble exactly. Users may replace or extend it freely.
- The model crate receives the prompt as `LlamaConfig::system_prompt`; it does
  no configuration-file I/O. This is a plain-text completion prefix, not a chat
  role/template. No built-in instructions are prepended. Missing separating
  newlines are added before the dynamic input; user content is otherwise
  preserved. Order: configured prompt, optional framed captured context (§6.8),
  recent committed text (§6.5). Context capture, clearing, and reset retain the
  configured prompt. PPM-only mode ignores instructions.
- Reads are bounded to **64 KiB**. Empty/whitespace-only, unreadable, invalid
  UTF-8, or oversized files produce a stderr warning and use the default without
  modifying the existing file. Tokenization errors also warn and use the default.
- Before warming the prefix, the worker validates the prompt (including its
  separator) against a maximum of **1024 model tokens**, further limited by the
  actual context capacity minus 1024 captured-context tokens, an 800-symbol
  byte-token worst-case typed window, and 32 tokens of headroom. Oversized
  instructions are rejected, never silently truncated. If even the default
  cannot fit the model context, ordinary model-startup failure falls back to PPM.

### Environment and data

| Variable | Effect |
|---|---|
| `KEYB_MODEL` | Path to a GGUF model instead of the managed default; never triggers a download. |
| `KEYB_NO_MODEL_DOWNLOAD` | Skip automatic provisioning; PPM only if the file is absent. |
| `KEYB_NO_PERSONALIZE` | Disable the user-vocabulary layer (§6.7). |
| `KEYB_ADAPTIVE` | Enable adaptive speed control (§5.2). |
| `KEYB_SHELL` | `layer` \| `x11` \| `window` — window frontend (§7.3). |
| `KEYB_INJECT` | `wayland` \| `wayland-vk` \| `x11` \| `buffer` — output backend (§8). |
| `KEYB_WIDTH` | OSK column width in logical px. |

- On first launch, the app downloads the default Qwen2.5-0.5B-Instruct
  Q4_K_M GGUF from the official Apache-2.0-licensed
  `Qwen/Qwen2.5-0.5B-Instruct-GGUF` repository. The revision, expected size,
  and SHA-256 are pinned in `keyb-app`; downloads are resumable (HTTP `Range`,
  with a safe restart when a server ignores it) and are only published to the
  final path after verification. The current artifact is 491,400,032 bytes with
  SHA-256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`.
  Download or load failure is never fatal: keyb falls back to PPM and still works.
- `$XDG_DATA_HOME/keyb/` — `models/` for the managed GGUF, and
  `user_model.v1.bin` for the user vocabulary (§6.7).
- `$XDG_STATE_HOME/keyb/last_context.txt` — debug dump of the exact bytes last
  promised to the model as an auxiliary context block (§6.8). Overwritten on
  each successful capture; not read back.
- **Privacy gaps against the original design**, both open:
  - the user-vocabulary store is a **binary** file, not the plain,
    hand-inspectable text the design promised (it is still user-deletable, and
    deleting it fully resets personalization);
  - the history toggle is the `KEYB_NO_PERSONALIZE` environment variable, **not
    surfaced in the main UI**. Since password-type fields cannot be detected
    reliably, the design's requirement that this toggle be visible rather than
    buried still stands and is unmet.
- **Session replay logs and persisted adaptive-speed state are not implemented.**
- The LLM input is the configured static prompt + optional captured context +
  recent committed text (§6.5–6.8). Instructions remain local; nothing leaves
  the machine after the one-time model download.

## 10. Performance budgets (mid-range 2020s laptop, CPU-only)

| Metric | Budget | Enforced? |
|---|---|---|
| Frame time (steady steering) | ≤ 8 ms p95 | No automated check |
| Pointer-to-photon latency | ≤ 2 frames | No automated check |
| LM forward pass | ≤ 50 ms p95 | No automated check |
| Distribution request → applied | ≤ 80 ms p95 | Yes — `real_model.rs` (skipped without a model) |
| Frames rendered with PPM-only sizing | < 5 % during steady steering | No automated check |
| RSS incl. model | ≤ 1.2 GB (Q4 0.5B) | No automated check |
| Cold start to steerable | ≤ 3 s (model mmap'd, preamble warmed at startup) | No automated check |

The oracle harness (§11) is the one budget that *is* continuously enforced, and
it does so end-to-end: the LLM oracle test paces simulated time to the wall
clock, so inference latency shows up directly as lost WPM.

## 11. Testing

What exists:

- **keyb-core**: unit tests for the invariants — children exactly partition their
  parent (to 1e-12), frame advance/climb-out round-trips exactly, dynamics are
  deterministic given (trace, seed), pan scrolls without zooming, upgrades
  animate while preserving the partition, release-commit gating, group-box
  descent and eviction, bounded expansion under a peaked model. These are
  hand-written table/scenario tests; **no property-testing framework is in use**
  (`keyb-core` has no dependencies at all), so the "property tests" and
  *recorded*-trace golden replay of the original plan are approximated by
  generated traces.
- **Marginalization**: `char_masses` is checked against brute-force summation
  over a synthetic vocabulary, plus healing-rollback and exact-token edge cases.
  Against the *real* tokenizer, `keyb-lm-llama/tests/real_model.rs` checks that
  upgrades are valid distributions, that the LLM beats uniform on obvious
  continuations, that healing preserves tokens spanning canonical boundaries,
  the §10 latency budget, and that a §6.8 context block moves predictions,
  survives tail-trimming, and restores preamble-only conditioning when cleared.
  It skips with a note when no model is present.
- **Throughput harness**: a simulated "oracle steerer" (`keyb_core::sim`) with
  configurable motor noise and reaction delay steers optimally toward target
  text and reports WPM. `keyb-app/tests/oracle.rs` gates ≥ 15 WPM on PPM (flat
  and grouped alphabets, plus a rare-letter case);
  `keyb-app/tests/oracle_llm.rs` gates ≥ 25 WPM on held-out English with the LLM,
  paced to the wall clock, and checks that a captured context block (§6.8)
  changes the empty-context distribution. `examples/bench_oracle.rs` and `examples/dbg_oracle.rs`
  are the manual sweep/debug entry points.
- **Injection**: `keyb-inject/tests/e2e_wayland.rs` round-trips commit + retract
  through a headless `sway` running `foot`, asserting the text actually lands in
  a real terminal. It skips when the binaries are absent.

Known gaps:

- **There is no CI.** Nothing runs the oracle harness on a pinned corpus per
  commit, and none of the §10 latency budgets are enforced as CI benchmarks with
  machine multipliers. Everything above runs only when someone types
  `cargo test --workspace`.
- **No X11 injection integration test** (Xvfb); backend 3 is verified by hand.

## 12. Milestones

1. **M1 — Steerable core (no LLM).** ✅ keyb-core + keyb-ui + PPM, text into an
   internal buffer. Exit met: oracle ≥ 15 WPM; a human can write a sentence.
2. **M2 — LLM prediction.** ✅ keyb-lm-llama with marginalization, branch cache,
   speculative expansion; smoothing + animated size upgrades. Exit met: oracle
   ≥ 25 WPM on held-out English (asserted when a model is present).
3. **M3 — System output.** ✅ keyb-inject backends 1–3, commit/retract ledger,
   layer-shell/override-redirect window behavior.
4. **M4 — Polish.** 🚧 In progress.
   - done: adaptive speed control (opt-in via `KEYB_ADAPTIVE`), alphabet group
     boxes, model auto-provisioning, persistent user vocabulary (§6.7),
     auxiliary context block (§6.8), collapse handle, reset control, and UI
     close control.
   - open: **action nodes** (§5.4), **settings UI** (§7.4), **config file and
     data-driven alphabets** (§9), **novice onboarding overlay**, **pointer
     capture** (§7.3), **UI-surfaced privacy toggle** (§9), packaging
     (Flatpak first).
5. **M5 — Hardening.** ⬜ Not started. Replay-driven fuzzing, crash-free 8-hour
   soak, WPM study with ≥ 3 fresh users; Windows/macOS injector spike
   (SendInput / CGEvent).

## 13. Risks

| Risk | Mitigation |
|---|---|
| Wayland compositors gate `input-method-v2`/`virtual-keyboard-v1` | Tiered backend fallback (§8), chosen per operation rather than at startup; document per-compositor support; Flatpak portal story tracked separately. |
| 0.5B model too weak → flat distributions → slow writing | Model is a `KEYB_MODEL` swap (1B–3B GGUF); oracle harness quantifies WPM-per-model so the trade is measured, not guessed. |
| Token-healing corner cases corrupt the marginalization | Marginalization is pure and tested against brute-force sums plus real-tokenizer edge cases (§11). |
| Retraction into arbitrary apps is inherently lossy | Capability-aware hysteresis and a bounded ledger that refuses what it can't verify (§8.3). The stronger mitigation once considered — confirm-before-retract outside the input-method backend — is **not implemented**. |
| f64 precision at deep zoom | Commit-independent frame advance/climb-out (§5.3) with exact-roundtrip tests. |
| A confident model deepens the tree until the arena fills and boxes vanish | Layered expansion caps and soft frame advance (§5.1, §5.3), each covered by a regression test. |
| vello API churn (pre-1.0) | Rendering isolated in keyb-ui behind a scene-description boundary; egui fallback renderer possible without touching core. |
| No CI: regressions land silently | Known gap (§11); the oracle harness exists and is cheap to wire up when CI is set up. |

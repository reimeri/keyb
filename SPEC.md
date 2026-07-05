# keyb — Implementation Specification

A mouse-driven, zooming on-screen text entry system in the lineage of Dasher, with a
local LLM as the probability model. Writing is continuous steering through a
landscape of nested boxes whose sizes are the model's predictive distribution;
no clicking, no discrete keys.

Status: draft v1 · Target platform: Linux desktop first (Wayland + X11), Windows/macOS later.

---

## 1. Goals

- **Fast text entry with a pointer only.** Target ≥ 25 WPM for a practiced user with
  an ordinary mouse; graceful degradation to trackball/trackpad/eye-tracker later.
- **LLM-driven prediction.** A small local model supplies the next-token
  distribution; long continuations surface naturally when probability × zoom makes
  them visually large. No cloud dependency, no network requirement at runtime.
- **Universal expressiveness.** Every possible string is reachable — smoothing
  guarantees no character ever has zero probability (novel names, codes, etc.).
- **System-wide output.** Committed text is injected into whatever application has
  focus, like a real on-screen keyboard — not just an internal text box.
- **Self-revealing UI.** Zero memorization required to start; the same steering
  skill deepens with practice with no ceiling imposed by clicking.

## 2. Non-goals (v1)

- Mobile/touch, eye-tracker, and single-switch input modes (the architecture must
  not preclude them — they are alternative sources of the same 2D steering signal).
- Cloud inference, user-model fine-tuning, multi-language UI (the alphabet and LM
  are pluggable; only English ships in v1).
- IME-grade integration with every toolkit. v1 targets the common paths listed in §8.

---

## 3. Tech stack

| Concern | Choice | Rationale |
|---|---|---|
| Language | **Rust** (stable, edition 2024) | Sustained 120 Hz rendering + real-time inference on worker threads with no GC pauses; first-class access to Wayland/X11 platform APIs; single static binary. |
| Windowing / input | **winit** | Mature cross-platform event loop; raw pointer events at device rate. |
| 2D rendering | **vello** (wgpu-based compute rasterizer) | The UI is thousands of nested rectangles and text glyphs under continuous zoom; vello renders resolution-independent vector scenes on the GPU with crisp text at any scale. |
| Text shaping/layout | **parley + swash** | Vello's companion stack; proper shaping, font fallback, subpixel-quality glyph rendering across zoom levels. |
| LLM inference | **llama.cpp** via the `llama-cpp-2` crate | GGUF models, CPU-first with optional Vulkan/CUDA offload, mature KV-cache API with **sequence branching** (needed for tree caches, §6.5). |
| Default model | **Qwen2.5-0.5B-Instruct, Q4_K_M GGUF** (~400 MB) | Sub-50 ms per forward pass on a mid-range CPU; far below chat quality bar but far above the entropy bar needed for box sizing. Swappable via config; SmolLM2-360M as low-end alternative. |
| Fallback char model | Order-4 **PPM** trained on user history + seed corpus, in-crate | Smoothing floor and offline/degraded mode; tiny and instant. |
| Text injection | Wayland `zwp_virtual_keyboard_v1` + `zwp_input_method_v2`; X11 XTEST (via `x11rb`); `wl_data_device`/clipboard as universal fallback | See §8. |
| Config/persistence | TOML via `serde` + `toml`; user data under `$XDG_DATA_HOME/keyb/` | Boring and inspectable. |
| Async/threading | std threads + `crossbeam-channel`; **no tokio** | The concurrency shape is a small fixed set of workers with bounded channels; an async runtime adds nothing. |

Why not the alternatives:

- **Tauri/Electron**: an OSK window must never take keyboard focus and needs
  layer-shell/override-redirect placement; web-view windows fight this. LLM and
  injection would live in a native sidecar anyway, leaving the web layer as pure
  overhead between the pointer and the renderer.
- **Qt/C++**: viable (original Dasher lineage), but we'd hand-roll the same
  worker/channel architecture with less safety, and QPainter/QML text under
  continuous zoom is weaker than a vector-first GPU renderer.
- **egui**: excellent for the settings window (and is used for it, §7.4), but its
  immediate-mode text/tessellation path is not built for a full-screen scene of
  continuously rescaling glyphs.

## 4. Repository layout

Cargo workspace:

```
keyb/
├─ SPEC.md
├─ Cargo.toml                # workspace
├─ crates/
│  ├─ keyb-core/             # §5, §6.1–6.3: geometry, dynamics, node tree, LM trait,
│  │                         #   char-trie marginalization, smoothing. No I/O, no GPU.
│  ├─ keyb-lm-llama/         # §6.4–6.6: llama.cpp backend, KV branch cache,
│  │                         #   speculative expansion worker.
│  ├─ keyb-lm-ppm/           # PPM fallback model (implements same trait).
│  ├─ keyb-ui/               # §7: winit window, vello scene building, input capture.
│  ├─ keyb-inject/           # §8: platform text output backends behind one trait.
│  └─ keyb-app/              # binary: wiring, config, session/replay logging.
└─ assets/                   # fonts, alphabet definitions, seed corpus for PPM
```

Dependency rule: `keyb-core` depends on nothing above std + serde. Everything else
depends on `keyb-core`. `keyb-ui` and `keyb-lm-*` never depend on each other —
they meet only through channels owned by `keyb-app`.

## 5. Core model (keyb-core)

### 5.1 Interval tree

Text space is the unit interval `[0,1)`. Each node owns a subinterval of its
parent proportional to its conditional probability; concatenating the labels on
the path from root to a point spells the text. This is arithmetic coding made
visible — `keyb-core` maintains the visible/needed portion of this infinite tree:

```rust
struct Node {
    label: Label,            // Char(char) | Token(String) | Action(EditOp)
    lo: f64, hi: f64,        // interval within the *root* frame (rebased on zoom, §5.3)
    state: NodeState,        // Collapsed | ExpandRequested | Expanded | Committed
    children: Vec<NodeId>,
    lm_handle: Option<BranchId>,  // KV-cache branch, if materialized (§6.5)
}
```

- **Expansion is lazy and screen-driven**: a node is expanded (children created,
  LM queried) when its projected screen height exceeds `expand_px` (default 40 px),
  and collapsed/evicted when below `evict_px` (default 2 px) or off-screen.
  Probability thresholds are never used directly — screen size *is*
  cumulative probability × zoom, which is the interface's native relevance measure.
- Node IDs come from a slab arena; the tree is rebuilt cheaply on eviction, never
  reference-counted.

### 5.2 Steering dynamics

Standard Dasher dynamics, kept deliberately boring because they're proven:

- Pointer position is read every frame in window coordinates and mapped into the
  model frame. **Horizontal displacement from the origin line controls zoom rate**
  (right = forward, left = backward/undo); **vertical position selects** which
  interval the view center drifts toward.
- Per frame: the view applies a zoom step `z = exp(k · dx · dt)` toward the point
  selected by `dy`, with the standard Dasher non-linearity that compresses extreme
  y-values so edge targets stay reachable.
- **Backward zoom is the only undo.** Zooming back across a commit boundary
  retracts committed text (§8.3).
- **Adaptive speed**: max zoom rate `k` rises slowly (multiplicative increments)
  while steering is smooth and falls immediately on direction reversals/jitter —
  a port of Dasher's auto speed control. Novice mode = low initial cap.
- **Pause/engage**: primary mouse button held = steering engaged; released =
  frozen. (Configurable to toggle mode for users who can't hold a button.)

Dynamics are a pure function `step(state, pointer, dt) -> state` — no I/O, fully
deterministic, unit-testable, and replayable from logged pointer traces (§11).

### 5.3 Numerics

Deep zooms exhaust f64 mantissa in the root frame. On each commit (§8.3) the
frame is **rebased**: the committed node's interval is rescaled to `[0,1)` and its
ancestors dropped. Invariant: the active view always spans ≥ 1e-6 of the current
frame. This caps error without arbitrary-precision arithmetic.

### 5.4 The alphabet is not just letters

Children of every node are drawn from an **alphabet definition** (TOML in
`assets/`): lowercase letters, space, digits and punctuation grouped under
expandable group boxes (as in classic Dasher), plus **action nodes** —
`DeleteWord`, `NewLine`, `CapitalizeNext` — which are ordinary steerable boxes
whose "probability" is a configured prior mixed into the distribution. Letter
*ordering is fixed forever* (alphabetical within groups); only sizes change.

## 6. Language model subsystem

### 6.1 The trait

```rust
trait LanguageModel: Send {
    /// Distribution over the next character/action, given committed text
    /// and the in-flight path. Must be exhaustive over the alphabet.
    fn predict(&mut self, ctx: &Context) -> CharDistribution;
    /// Hint that `ctx` was abandoned (branch eviction).
    fn release(&mut self, ctx: &Context);
}
```

`keyb-core` consumes only *character-level* distributions over the alphabet.
Token thinking is entirely encapsulated in `keyb-lm-llama`.

### 6.2 Token → character marginalization

The LLM predicts subword tokens; the UI must stay character-honest (a user may
steer through the middle of what the model tokenized as one unit). After each
forward pass:

1. Take the logits over the full vocab; softmax → `p(token)`.
2. Filter to tokens consistent with the current intra-token prefix (token
   healing: if the user has partially spelled into a token, only tokens whose
   bytes extend that prefix survive; renormalize).
3. Build a **character trie**: `p(next char c) = Σ p(token t) for t whose next
   byte sequence starts with c` (UTF-8 aware; multi-byte chars grouped).
4. Cache the trie on the node. Descending *within* the trie (steering deeper into
   a long token like `" strawberry"`) costs **zero additional forward passes** —
   this is what makes token-level prediction feel like multi-character jumps.

A new forward pass is needed only when the user's path exhausts the trie (crosses
a token boundary).

### 6.3 Smoothing

```
p_final(c) = (1 − λ) · p_llm(c) + λ · p_ppm(c),   then floor at p_min and renormalize
```

- `λ = 0.05` default; `p_min = 1e-4` per alphabet symbol.
- Action nodes get configured priors mixed in at this stage.
- If the LLM worker misses its deadline (§6.6), the frame uses `p_ppm` alone and
  upgrades in place when the real distribution arrives — sizes animate to new
  values over ~120 ms so boxes never pop.

### 6.4 Inference worker

One dedicated thread owns the llama.cpp context. Protocol over bounded channels:

```
UI → LM : ExpandRequest { node_id, context, priority }   // priority = projected screen area
UI → LM : Evict { branch_id }
LM → UI : Distribution { node_id, char_trie, branch_id }
```

Requests sit in a priority queue keyed by screen area, **re-prioritized every
frame** and cancellable — steering away from a branch cancels its pending work.

### 6.5 KV-cache branching

llama.cpp sequence IDs implement the tree cache:

- Each materialized branch (node whose distribution came from a real forward
  pass) holds a sequence ID; expanding a child = `kv_cache_seq_cp` (copy-on-write
  prefix reuse) + decode of the new token(s).
- Budget: `max_branches = 32` sequences. Eviction is LRU weighted by screen area;
  evicted branches that get re-entered are recomputed from the committed-text
  prefix (which always stays resident as sequence 0).
- On commit + rebase (§5.3), all branches not under the committed node are freed.

### 6.6 Speculative expansion

The user steers at seconds-per-word; the model runs at tens of ms per token —
spend the surplus so zooming never stalls:

- After servicing the visible frontier, the worker speculatively expands the
  **top-k (k=4) children by screen area** of the node the view center is inside,
  one token deep, batched in a single decode call.
- Deadline discipline: the worker never blocks the UI; the UI never waits. A node
  whose distribution hasn't arrived renders with PPM sizing (§6.3).
- Perf budget: forward pass ≤ 50 ms p95 on 4 CPU threads; full pipeline
  (request → trie on node) ≤ 80 ms p95.

## 7. Rendering (keyb-ui)

### 7.1 Scene

- Rebuild the vello scene every frame from the visible tree (immediate-mode over
  retained tree). Visible node count is bounded (~200) by the expand/evict
  thresholds, so scene building is trivially within budget.
- Boxes: nested rounded rects, alternating hue by depth (classic Dasher scheme,
  colorblind-safe palette); the box under the view center gets a subtle outline.
- **Text LOD**: label glyphs fade in when box height ≥ 18 px, scale with the box
  between 18–120 px, clamp at a max glyph size beyond. Long token/word labels
  render once legible; below threshold boxes are color-only.
- Committed text ribbon along the left edge (most recent ~60 chars), plus a
  faint full-sentence echo strip along the top.

### 7.2 Frame loop

`winit` redraw-on-vsync; per-frame: read pointer → `keyb_core::step()` → drain
LM channel (apply arrived distributions, animate sizes) → emit expand/evict
requests → build scene → render. **Frame budget 8 ms** (comfortable at 120 Hz,
safe at 60 Hz).

### 7.3 Window behavior

- Wayland: **wlr-layer-shell** surface (overlay layer, anchored/user-positioned),
  `keyboard_interactivity: none` — the window can *never* steal keyboard focus.
- X11: override-redirect window with `_NET_WM_WINDOW_TYPE_DOCK` +
  no-input-focus hints (`WM_HINTS.input = False`).
- The pointer is *captured* while steering is engaged (button held): pointer-lock
  (relative motion) so the cursor can't wander off the OSK window mid-word.

### 7.4 Settings

Separate normal window using `egui` (via `egui-wgpu` on the same device):
model selection, speed caps, alphabet, injection backend, telemetry opt-in.
Never visible during typing.

## 8. Text output (keyb-inject)

```rust
trait Injector: Send {
    fn commit(&mut self, text: &str) -> Result<()>;
    fn retract(&mut self, chars: usize) -> Result<()>;  // undo support
    fn capabilities(&self) -> Caps;                     // true retraction? surrounding text?
}
```

Backends, auto-detected in order:

1. **Wayland `zwp_input_method_v2`** (preferred where the compositor allows it):
   real preedit + commit + `delete_surrounding_text` → true retraction.
2. **Wayland `zwp_virtual_keyboard_v1`**: synthesizes key events from a dynamic
   keymap; retraction = synthesized backspaces.
3. **X11 XTEST**: synthesized key events; retraction = backspaces.
4. **Clipboard-assist fallback**: types via paste; last resort, off by default.

### 8.3 Commit semantics

- Text is committed when its node's interval fully contains the view frame
  (i.e., the user has zoomed entirely inside it) **and** has stayed contained
  for 150 ms (debounce against overshoot).
- A rolling **retraction ledger** of the last 32 commits is kept; backward zoom
  across a commit boundary pops the ledger and calls `retract()`. Injectors
  without true retraction use backspace synthesis; if capabilities report
  unreliable retraction (e.g., target app unknown), the UI requires a deeper
  back-zoom (hysteresis) before retracting, to avoid mangling text.

## 9. Configuration & persistence

- `$XDG_CONFIG_HOME/keyb/config.toml` — model path, threads, speed, alphabet,
  injection preferences, novice mode.
- `$XDG_DATA_HOME/keyb/` — PPM user history (plain text, user-deletable; **an
  explicit privacy toggle disables history entirely**, and password-type fields
  can't be detected reliably so the toggle is surfaced in the main UI, not buried),
  session replay logs (opt-in), adaptive-speed state.
- The LLM context is primed with a short system preamble + recent committed text
  only. Nothing leaves the machine.

## 10. Performance budgets (mid-range 2020s laptop, CPU-only)

| Metric | Budget |
|---|---|
| Frame time (steady steering) | ≤ 8 ms p95 |
| Pointer-to-photon latency | ≤ 2 frames |
| LM forward pass | ≤ 50 ms p95 |
| Distribution request → applied | ≤ 80 ms p95 |
| Frames rendered with PPM-only sizing | < 5 % during steady steering |
| RSS incl. model | ≤ 1.2 GB (Q4 0.5B) |
| Cold start to steerable | ≤ 3 s (model mmap'd, first token warmed in background) |

## 11. Testing

- **keyb-core**: property tests — intervals always partition `[0,1)`; rebase
  preserves relative geometry to 1e-12; dynamics are deterministic given
  (trace, seed). Golden replay: recorded pointer traces must reproduce identical
  committed text across refactors.
- **Marginalization**: exhaustive check that char-trie masses equal summed token
  masses for random contexts; token-healing edge cases (multi-byte UTF-8 split
  across tokens) as table tests against the real tokenizer.
- **Throughput harness**: a simulated "oracle steerer" (steers optimally toward
  target text with configurable motor noise/reaction delay) produces WPM and
  error-rate benchmarks per model/config — this is the regression metric for any
  change to dynamics or prediction. Run in CI on a pinned corpus.
- **Injection**: integration tests against a headless compositor
  (`sway --headless` / Xvfb) asserting round-trip of commit + retract into a
  test client.
- Latency budgets in §10 enforced as CI benchmarks with generous CI-machine
  multipliers.

## 12. Milestones

1. **M1 — Steerable core (no LLM).** keyb-core + keyb-ui + PPM only, text into an
   internal buffer shown on screen. Exit: oracle harness ≥ 15 WPM; a human can
   write a sentence.
2. **M2 — LLM prediction.** keyb-lm-llama with marginalization, branch cache,
   speculative expansion; smoothing + animated size upgrades. Exit: oracle ≥ 25 WPM
   on held-out English; p95 budgets met.
3. **M3 — System output.** keyb-inject backends 1–3, commit/retract ledger,
   layer-shell/override-redirect window behavior. Exit: dictate text into
   Firefox + a GTK app + a terminal on both Wayland and X11.
4. **M4 — Polish.** Adaptive speed tuning, novice onboarding overlay, settings UI,
   alphabet groups + action nodes, config persistence, packaging (Flatpak first).
5. **M5 — Hardening.** Replay-driven fuzzing, crash-free 8-hour soak, WPM study
   with ≥ 3 fresh users; Windows/macOS injector spike (SendInput / CGEvent).

## 13. Risks

| Risk | Mitigation |
|---|---|
| Wayland compositors gate `input-method-v2`/`virtual-keyboard-v1` | Three-tier backend fallback (§8); document per-compositor support; Flatpak portal story tracked separately. |
| 0.5B model too weak → flat distributions → slow writing | Model is a config swap (1B–3B GGUF); oracle harness quantifies WPM-per-model so the trade is measured, not guessed. |
| Token-healing corner cases corrupt the trie | Marginalization is pure and exhaustively property-tested against the real tokenizer (§11). |
| Retraction into arbitrary apps is inherently lossy | Capability-aware hysteresis (§8.3); conservative default: confirm-before-retract outside input-method backend. |
| f64 precision at deep zoom | Rebase-on-commit invariant (§5.3) with property tests. |
| vello API churn (pre-1.0) | Rendering isolated in keyb-ui behind a scene-description boundary; egui fallback renderer possible without touching core. |

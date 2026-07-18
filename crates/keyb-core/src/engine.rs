//! Interval tree + steering dynamics + commit/retract (SPEC §5).
//!
//! Text space is the unit interval [0,1) of the *current frame* — the node
//! the tree is rooted at. The frame and the committed text move
//! independently: commits/retractions only shift a debounced watermark in the
//! text buffer and never touch geometry, so rendering stays continuous. The
//! frame itself moves rarely: forward ("advance") when the view gets tiny
//! relative to it, to keep f64 away from deep-zoom cancellation, and backward
//! ("climb out") when the view escapes it — an exact reconstruction from the
//! ledger, invisible on screen.
//!
//! The alphabet may be *grouped* (SPEC §5.4): rare runs (capitals, digits,
//! symbols) sit inside expandable group boxes. A node therefore carries a
//! `label` (the character it commits, or the group it opens) and an `opens`
//! layout id naming the choices its children are drawn from. A group box adds
//! an intermediate tree level on the path to a grouped leaf but leaves the
//! final leaf's interval untouched — so ungrouped lowercase steering is
//! bit-for-bit what it was before grouping existed.

use crate::alphabet::{Alphabet, GroupColor, Item};
use crate::lm::LanguageModel;

/// Per-frame steering input, already normalized by the front end.
#[derive(Clone, Copy, Debug)]
pub struct PointerInput {
    /// Zoom drive in [-1, 1]: positive zooms in (writes), negative zooms out
    /// (undoes). 0 holds.
    pub dx: f64,
    /// Selection point as a fraction of the view (0 = top, 1 = bottom). May
    /// slightly exceed [0,1] to let edge targets be steered at.
    pub y: f64,
    /// False freezes all motion (button released).
    pub engaged: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct EngineParams {
    /// Zoom rate at full deflection, in bits/second (view halves per 1/rate s).
    pub max_bits_per_sec: f64,
    /// Vertical pan rate as a multiple of the zoom rate: the point under the
    /// pointer drifts toward the view center at `pan_gain * speed` bits/s of
    /// gap-halving. Pan is what makes pure vertical pointer motion scroll.
    pub pan_gain: f64,
    /// Expand a node once its projected height exceeds this (px).
    pub expand_px: f64,
    /// Evict children once a node's projected height falls below this (px).
    pub evict_px: f64,
    /// A node must fully contain the view this long before committing (s).
    pub commit_debounce: f64,
    /// Advance the frame once the view is narrower than this fraction of it.
    pub rebase_view_w: f64,
    /// Advance the frame once the crosshair is contained this many tree levels
    /// below the frame root, regardless of the view's absolute width. Keeps the
    /// materialized tree shallow so a confident (peaked) model can't drive the
    /// speculative frontier into the `max_depth` wall and strand the crosshair
    /// in a box with no children ("no boxes shown"). Must be < `max_depth`.
    pub soft_advance_depth: usize,
    /// Expansion depth limit below the current frame root (safety net; the
    /// pixel thresholds are the real gate).
    pub max_depth: usize,
    /// Arena size that triggers compaction (dead-node garbage collection).
    pub max_nodes: usize,
    /// Async distribution upgrades animate box sizes toward the new partition
    /// over roughly this long (SPEC §6.3) so boxes never pop.
    pub upgrade_anim_secs: f64,
    /// Retraction hysteresis: the view may exceed the deepest committed node
    /// by this fraction of its width before a retraction is considered.
    /// keyb-app widens this when the active injector lacks true retraction
    /// (SPEC §8.3) so backspace synthesis into unknown apps needs a firmer
    /// back-zoom.
    pub retract_eps: f64,
    /// Release-commit gate: when steering disengages (button released), any
    /// crosshair-path box taller than this fraction of the view is committed
    /// on the spot, even though it never fully engulfed the view. Lets a user
    /// steer at a target and let go rather than zoom all the way in — handy
    /// for landing a space/period to end a sentence. Boxes below the fraction
    /// stay uncommitted, so pausing mid-word never dumps tiny speculative
    /// descendants into the buffer.
    pub commit_release_frac: f64,
    /// Auto speed control (SPEC §5.2): when true, `speed_bits` rises slowly
    /// while steering is smooth and drops immediately on jitter or a zoom
    /// reversal. Off by default so the oracle harness and unit tests drive
    /// `speed_bits` directly; keyb-app turns it on from config.
    pub adaptive_speed: bool,
    /// Floor for the adaptive speed cap (also the novice starting point).
    pub min_bits_per_sec: f64,
    /// Multiplicative rise per second while steering smoothly forward.
    pub speed_rise_per_sec: f64,
    /// Fraction the cap is cut to on a detected reversal/jitter event.
    pub speed_drop: f64,
}

impl Default for EngineParams {
    fn default() -> Self {
        EngineParams {
            max_bits_per_sec: 4.5,
            pan_gain: 0.8,
            expand_px: 40.0,
            evict_px: 2.0,
            commit_debounce: 0.15,
            rebase_view_w: 1e-6,
            soft_advance_depth: 16,
            max_depth: 64,
            max_nodes: 12000,
            upgrade_anim_secs: 0.12,
            retract_eps: 0.02,
            commit_release_frac: 0.5,
            adaptive_speed: false,
            min_bits_per_sec: 1.5,
            speed_rise_per_sec: 0.35,
            speed_drop: 0.6,
        }
    }
}

/// A change to the committed text, for system output (SPEC §8). Drained by
/// the front end once per frame and forwarded to the injector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TextEvent {
    /// Text was appended to the committed buffer.
    Commit(String),
    /// The last committed symbol was retracted (always one symbol; its
    /// decoded char count is up to the ledger on the injection side).
    Retract(String),
}

/// What a tree node is: a committed leaf character, or a group box that opens
/// into its members. Group nodes add a level on the path to a grouped leaf but
/// never enter the committed text.
#[derive(Clone, Copy, Debug)]
enum NodeLabel {
    Char(u8),
    Group(u32),
}

struct Node {
    label: NodeLabel,
    /// Layout node id whose items this node's children are drawn from (a leaf
    /// opens into the root layout — the next character; a group opens into
    /// itself).
    opens: u32,
    lo: f64,
    hi: f64,
    /// Depth below the current frame root (shifts when the frame moves).
    depth: usize,
    parent: usize,
    children: Vec<usize>,
    expanded: bool,
    alive: bool,
    /// Child partition an async upgrade is animating toward:
    /// `anim_target[k]..anim_target[k+1]` is child `k`'s destination
    /// (`anim_target[0] == lo`, last == hi). Frame coordinates, so every
    /// affine tree move remaps it alongside `lo`/`hi`.
    anim_target: Option<Box<[f64]>>,
}

impl Node {
    fn as_char(&self) -> Option<u8> {
        match self.label {
            NodeLabel::Char(s) => Some(s),
            NodeLabel::Group(_) => None,
        }
    }
    fn is_group(&self) -> bool {
        matches!(self.label, NodeLabel::Group(_))
    }
}

/// One frame-absorbed tree level: the parent's complete child partition
/// (`cuts[i]..cuts[i+1]` is item `i`), the layout node the parent opened into
/// (so the sibling labels can be rebuilt), which child the frame descended
/// into, and whether that child was a leaf (i.e. consumed a committed
/// character). Climbing back out re-lays the parent exactly as it was on
/// screen, immune to LM drift.
#[derive(Clone, Debug)]
struct LedgerEntry {
    cuts: Box<[f64]>,
    opens: u32,
    child_idx: usize,
    was_char: bool,
}

/// A renderable box: vertical extent as fractions of the canvas (may exceed
/// [0,1]; the renderer clips).
#[derive(Clone, Copy, Debug)]
pub struct BoxView {
    /// Leaf symbol for a character box (unused for group boxes).
    pub sym: u8,
    /// Glyph to draw for this box (character label, or the group's mark).
    pub label: char,
    /// Set for a group box; drives its distinct, popping color (SPEC §5.4).
    pub group: Option<GroupColor>,
    /// Arena index of this node. Lets the renderer relate boxes to their
    /// children (via `parent`) without knowing the tree layout.
    pub id: usize,
    /// Arena index of this node's parent. A parent not present in the visible
    /// set (ROOT or culled) is harmlessly ignored by the renderer.
    pub parent: usize,
    /// Absolute depth from the document root (stable across frame moves, so
    /// palette parity never flips on screen).
    pub depth: usize,
    pub y0: f64,
    pub y1: f64,
    /// On the crosshair path (contains the view center): this is what will be
    /// written if the user keeps steering here.
    pub on_path: bool,
}

pub struct Engine {
    pub alphabet: Alphabet,
    lm: Box<dyn LanguageModel>,
    pub params: EngineParams,
    nodes: Vec<Node>,
    view_lo: f64,
    view_hi: f64,
    /// Full committed text (leaf symbols). The first `frame_chars` are
    /// absorbed into the frame; the rest (the "tail") are nodes in the tree.
    committed: Vec<u8>,
    ledger: Vec<LedgerEntry>,
    /// Committed characters absorbed into the frame. Distinct from
    /// `ledger.len()` (tree levels) because a grouped character is absorbed as
    /// two levels (group box, then leaf).
    frame_chars: usize,
    commit_candidate: Option<u8>, // symbol of first uncommitted contained node
    commit_timer: f64,
    outside_timer: f64,
    /// Engaged state of the previous `step`, to catch the release edge.
    was_engaged: bool,
    /// Previous frame's selection point and zoom sign, for auto speed control.
    prev_y: f64,
    prev_dx_sign: i8,
    viewport_px: f64,
    /// Current speed cap in bits/s; UI may adjust, sim sets directly.
    pub speed_bits: f64,
    /// Committed-text changes since the last `drain_text_events` call.
    events: Vec<TextEvent>,
}

const ROOT: usize = 0;
const MIN_VIEW_W: f64 = 1e-9;
const MAX_VIEW_W: f64 = 4.0;

/// Copy the alive subtree at `src_idx` into `dst`, mapping coordinates
/// `x -> offset + x * scale` and re-rooting depths at `depth`. Returns the
/// new index of the subtree root.
fn copy_subtree(
    src: &[Node],
    src_idx: usize,
    dst: &mut Vec<Node>,
    parent: usize,
    depth: usize,
    offset: f64,
    scale: f64,
) -> usize {
    let n = &src[src_idx];
    let idx = dst.len();
    dst.push(Node {
        label: n.label,
        opens: n.opens,
        lo: offset + n.lo * scale,
        hi: offset + n.hi * scale,
        depth,
        parent,
        children: Vec::new(),
        expanded: n.expanded,
        alive: true,
        anim_target: n
            .anim_target
            .as_ref()
            .map(|cuts| cuts.iter().map(|&c| offset + c * scale).collect()),
    });
    let mut kids = Vec::with_capacity(n.children.len());
    for &c in &n.children {
        if src[c].alive {
            kids.push(copy_subtree(src, c, dst, idx, depth + 1, offset, scale));
        }
    }
    dst[idx].children = kids;
    idx
}

impl Engine {
    pub fn new(alphabet: Alphabet, lm: Box<dyn LanguageModel>, params: EngineParams) -> Self {
        assert_eq!(lm.alphabet_len(), alphabet.len());
        let root = alphabet.root_layout();
        Engine {
            speed_bits: if params.adaptive_speed {
                params.min_bits_per_sec
            } else {
                params.max_bits_per_sec
            },
            alphabet,
            lm,
            params,
            nodes: vec![root_node(root)],
            view_lo: 0.0,
            view_hi: 1.0,
            committed: Vec::new(),
            ledger: Vec::new(),
            frame_chars: 0,
            commit_candidate: None,
            commit_timer: 0.0,
            outside_timer: 0.0,
            was_engaged: false,
            prev_y: 0.5,
            prev_dx_sign: 0,
            viewport_px: 800.0,
            events: Vec::new(),
        }
    }

    /// Return the model to its just-constructed state — tree back to the root,
    /// committed buffer and prediction context wiped — while keeping the
    /// alphabet, language model, params, and current viewport. The LM's adaptive
    /// state is rewound by replaying `unlearn` over the committed buffer in
    /// reverse commit order (the exact inverse of the commit path), so
    /// predictions return to baseline. No `TextEvent` is emitted: text already
    /// injected into the target application is deliberately left untouched.
    pub fn reset(&mut self) {
        while let Some(sym) = self.committed.pop() {
            self.lm.unlearn(&self.committed, sym);
        }
        self.nodes = vec![root_node(self.alphabet.root_layout())];
        self.view_lo = 0.0;
        self.view_hi = 1.0;
        self.ledger.clear();
        self.frame_chars = 0;
        self.commit_candidate = None;
        self.commit_timer = 0.0;
        self.outside_timer = 0.0;
        self.was_engaged = false;
        self.prev_dx_sign = 0;
        self.speed_bits = if self.params.adaptive_speed {
            self.params.min_bits_per_sec
        } else {
            self.params.max_bits_per_sec
        };
        self.events.clear();
    }

    pub fn set_viewport_px(&mut self, px: f64) {
        self.viewport_px = px.max(1.0);
    }

    pub fn committed_syms(&self) -> &[u8] {
        &self.committed
    }

    pub fn committed_text(&self) -> String {
        self.alphabet.decode(&self.committed)
    }

    pub fn view(&self) -> (f64, f64) {
        (self.view_lo, self.view_hi)
    }

    /// Committed-text changes since the last call, in order (SPEC §8). The
    /// front end forwards these to the injection worker every frame.
    pub fn drain_text_events(&mut self) -> Vec<TextEvent> {
        std::mem::take(&mut self.events)
    }

    fn tail_len(&self) -> usize {
        self.committed.len() - self.frame_chars
    }

    /// Advance the model by `dt` seconds under `input`.
    pub fn step(&mut self, input: PointerInput, dt: f64) {
        let dt = dt.clamp(0.0, 0.1);
        if input.engaged {
            if self.params.adaptive_speed {
                self.adapt_speed(input, dt);
            }
            self.zoom(input, dt);
        }
        self.prev_y = input.y;
        self.apply_upgrades();
        self.animate_partitions(dt);
        self.climb_out_if_outside();
        self.maintain_tree();
        self.send_frame_hint();
        // A genuine button release (engaged high → low) lands the letters the
        // crosshair is clearly over, ahead of the containment debounce. The
        // frontends also drop `engaged` as a fail-safe when the pointer escapes
        // the column mid-drag; the release-commit fraction gate keeps that from
        // committing anything the user hadn't already zoomed well into.
        if self.was_engaged && !input.engaged {
            self.commit_under_crosshair();
        }
        self.was_engaged = input.engaged;
        self.update_commit(dt, input.engaged);
        self.advance_frame_if_deep();
        // Advancing re-roots the frame; expand the freshly-exposed frontier now
        // so the presented tree is never a frame behind (which would flash a
        // childless box the user just zoomed into).
        self.maintain_tree();
    }

    /// Auto speed control (SPEC §5.2): a port of Dasher's — reward smooth,
    /// forward steering with slow multiplicative growth of the cap, and punish
    /// jitter (a large lateral jump) or a zoom-direction reversal with an
    /// immediate cut. The cap floats within `[min_bits, max_bits]`.
    fn adapt_speed(&mut self, input: PointerInput, dt: f64) {
        let dx_sign = if input.dx > 0.05 {
            1
        } else if input.dx < -0.05 {
            -1
        } else {
            0
        };
        // Jitter proxy: lateral speed well above what smooth aiming needs.
        let jitter = (input.y - self.prev_y).abs() > 0.25;
        let reversed = self.prev_dx_sign != 0 && dx_sign != 0 && dx_sign != self.prev_dx_sign;
        if jitter || reversed {
            self.speed_bits =
                (self.speed_bits * self.params.speed_drop).max(self.params.min_bits_per_sec);
        } else if dx_sign > 0 {
            let grow = 1.0 + self.params.speed_rise_per_sec * dt;
            self.speed_bits = (self.speed_bits * grow).min(self.params.max_bits_per_sec);
        }
        if dx_sign != 0 {
            self.prev_dx_sign = dx_sign;
        }
    }

    /// Drain async LM upgrades (SPEC §6.3): a node expanded with fallback
    /// sizing gets a target partition to animate toward. Nodes that vanished
    /// (evicted, above the frame) are skipped — a re-expansion will hit the
    /// LM's cache synchronously instead. A single character context can size
    /// several boxes: the character-position node *and* any expanded group
    /// boxes beneath it (which partition the same distribution over their
    /// members), so all of them are re-laid out.
    fn apply_upgrades(&mut self) {
        for (ctx, dist) in self.lm.poll_upgrades() {
            debug_assert_eq!(dist.len(), self.alphabet.len());
            let Some(p) = self.node_for_context(&ctx) else {
                continue;
            };
            let mut stack = vec![p];
            while let Some(n) = stack.pop() {
                self.repartition_node(n, &dist);
                for &c in &self.nodes[n].children.clone() {
                    if self.nodes[c].alive && self.nodes[c].expanded && self.nodes[c].is_group() {
                        stack.push(c);
                    }
                }
            }
        }
    }

    /// Set `node`'s animation target from a fresh leaf distribution, mapping it
    /// through the node's layout so groups get their aggregate mass.
    fn repartition_node(&mut self, i: usize, dist: &[f64]) {
        if !self.nodes[i].alive || !self.nodes[i].expanded {
            return;
        }
        let opens = self.nodes[i].opens;
        let weights = self.alphabet.item_weights(opens, dist);
        if weights.len() != self.nodes[i].children.len() {
            return;
        }
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return;
        }
        let (lo, hi) = (self.nodes[i].lo, self.nodes[i].hi);
        let w = hi - lo;
        let mut cuts = Vec::with_capacity(weights.len() + 1);
        let mut acc = lo;
        cuts.push(lo);
        for &wt in &weights[..weights.len() - 1] {
            acc += wt / total * w;
            cuts.push(acc);
        }
        cuts.push(hi);
        self.nodes[i].anim_target = Some(cuts.into());
    }

    /// Walk `context` (absolute: document start) down from the frame root to
    /// the character-position node it names (a leaf, or ROOT for the empty
    /// relative context).
    fn node_for_context(&self, ctx: &[u8]) -> Option<usize> {
        let prefix = &self.committed[..self.frame_chars];
        if !ctx.starts_with(prefix) {
            return None; // frame advanced past this context, or text retracted
        }
        let mut cur = ROOT;
        for &c in &ctx[prefix.len()..] {
            cur = self.descend_one_char(cur, c)?;
        }
        Some(cur)
    }

    /// Descend from `start` through one character's worth of layout levels
    /// (possibly a group box, then its member leaf), returning the leaf node.
    /// `None` if any needed level isn't expanded/alive.
    fn descend_one_char(&self, start: usize, c: u8) -> Option<usize> {
        let mut cur = start;
        loop {
            if !self.nodes[cur].expanded {
                return None;
            }
            let opens = self.nodes[cur].opens;
            let (k, is_leaf) = self.alphabet.item_leading_to(opens, c)?;
            let child = *self.nodes[cur].children.get(k)?;
            if !self.nodes[child].alive {
                return None;
            }
            cur = child;
            if is_leaf {
                return Some(cur);
            }
        }
    }

    /// Move animating partitions toward their targets (exponential approach,
    /// ~95% converged in `upgrade_anim_secs`) and re-lay child subtrees
    /// affinely into the new cuts. Runs even while disengaged so distributions
    /// arriving during a pause still settle smoothly.
    fn animate_partitions(&mut self, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        // Time constant: 3τ ≈ full animation window.
        let tau = (self.params.upgrade_anim_secs / 3.0).max(1e-3);
        let blend = 1.0 - (-dt / tau).exp();
        for i in 0..self.nodes.len() {
            if !self.nodes[i].alive || self.nodes[i].anim_target.is_none() {
                continue;
            }
            let target = self.nodes[i].anim_target.clone().expect("checked above");
            let w = self.nodes[i].hi - self.nodes[i].lo;
            let kids = self.nodes[i].children.clone();
            debug_assert_eq!(kids.len() + 1, target.len());
            let mut done = true;
            let mut cursor = self.nodes[i].lo;
            for (k, &c) in kids.iter().enumerate() {
                let (old_lo, old_hi) = (self.nodes[c].lo, self.nodes[c].hi);
                let new_hi = if k + 1 == kids.len() {
                    self.nodes[i].hi
                } else {
                    let goal = target[k + 1];
                    let stepped = old_hi + (goal - old_hi) * blend;
                    let v = if (goal - stepped).abs() > 1e-4 * w {
                        done = false;
                        stepped
                    } else {
                        goal
                    };
                    v.max(cursor) // snap epsilon must never disorder cuts
                };
                self.remap_subtree(c, old_lo, old_hi, cursor, new_hi);
                cursor = new_hi;
            }
            if done {
                self.nodes[i].anim_target = None;
            }
        }
    }

    /// Affinely map the subtree at `i` from [old_lo, old_hi] to [lo, hi].
    fn remap_subtree(&mut self, i: usize, old_lo: f64, old_hi: f64, lo: f64, hi: f64) {
        let scale = (hi - lo) / (old_hi - old_lo).max(f64::MIN_POSITIVE);
        let offset = lo - old_lo * scale;
        let mut stack = vec![i];
        while let Some(n) = stack.pop() {
            let node = &mut self.nodes[n];
            node.lo = offset + node.lo * scale;
            node.hi = offset + node.hi * scale;
            if let Some(cuts) = node.anim_target.as_mut() {
                for c in cuts.iter_mut() {
                    *c = offset + *c * scale;
                }
            }
            stack.extend(node.children.iter().copied());
        }
        // Kill rounding drift at the seams: the subtree root sits exactly.
        self.nodes[i].lo = lo;
        self.nodes[i].hi = hi;
    }

    /// Tell the LM where the user is steering (deepest expanded node under
    /// the view center) so it can re-prioritize and speculate (SPEC §6.6).
    fn send_frame_hint(&mut self) {
        let path = self.center_path();
        let deepest_expanded = path
            .iter()
            .rev()
            .find(|&&i| self.nodes[i].expanded)
            .copied()
            .unwrap_or(ROOT);
        let ctx = self.context_of(deepest_expanded);
        self.lm.frame_hint(&ctx);
    }

    fn zoom(&mut self, input: PointerInput, dt: f64) {
        let w = self.view_hi - self.view_lo;
        let dx = input.dx.clamp(-1.0, 1.0);
        let y = input.y.clamp(-0.5, 1.5);
        let new_w = (w * (-self.speed_bits * dx * dt).exp2()).clamp(MIN_VIEW_W, MAX_VIEW_W);
        // Dasher dynamics: the point under the pointer drifts toward the view
        // center (vertical pan) independently of the zoom axis, so pure
        // vertical pointer motion scrolls without zooming.
        let p = self.view_lo + y * w;
        let y_new = 0.5 + (y - 0.5) * (-(self.params.pan_gain * self.speed_bits * dt)).exp2();
        self.view_lo = p - y_new * new_w;
        self.view_hi = self.view_lo + new_w;
    }

    /// Keep the view inside the frame. Escaping upward pops ledger levels —
    /// the stored partition is re-laid out and the current tree transplanted
    /// into its slot, so nothing visible moves. Text is *not* retracted here;
    /// that is `update_commit`'s debounced job. With an empty ledger the view
    /// clamps to the document root instead.
    fn climb_out_if_outside(&mut self) {
        loop {
            if self.view_lo >= 0.0 && self.view_hi <= 1.0 {
                return;
            }
            if self.ledger.is_empty() {
                let w = (self.view_hi - self.view_lo).clamp(MIN_VIEW_W, 1.0);
                let lo = self.view_lo.clamp(0.0, 1.0 - w);
                self.view_lo = lo;
                self.view_hi = lo + w;
                return;
            }
            self.climb_out_once();
        }
    }

    fn climb_out_once(&mut self) {
        let e = self.ledger.pop().expect("caller checked non-empty");
        let opens = e.opens;
        let items: Vec<Item> = self.alphabet.items(opens).to_vec();
        let k = e.child_idx;
        let (c_lo, c_hi) = (e.cuts[k], e.cuts[k + 1]);
        let cw = c_hi - c_lo;
        let mut dst: Vec<Node> = Vec::with_capacity(self.nodes.len() + items.len() + 1);
        dst.push(Node {
            label: NodeLabel::Char(0),
            opens,
            lo: 0.0,
            hi: 1.0,
            depth: 0,
            parent: ROOT,
            children: Vec::new(),
            expanded: true,
            alive: true,
            anim_target: None,
        });
        let mut kids = Vec::with_capacity(items.len());
        for (i, &item) in items.iter().enumerate() {
            if i == k {
                // Transplant the current tree into the descended slot. The old
                // root's label is a placeholder when it was created by a climb
                // (a climb root's identity lives one ledger level up), so
                // restamp label/opens from the authoritative layout item.
                let idx = copy_subtree(&self.nodes, ROOT, &mut dst, ROOT, 1, c_lo, cw);
                let (label, c_opens) = label_and_opens(&self.alphabet, item);
                dst[idx].label = label;
                dst[idx].opens = c_opens;
                dst[idx].lo = c_lo;
                dst[idx].hi = c_hi;
                kids.push(idx);
            } else {
                let (label, c_opens) = label_and_opens(&self.alphabet, item);
                let idx = dst.len();
                dst.push(Node {
                    label,
                    opens: c_opens,
                    lo: e.cuts[i],
                    hi: e.cuts[i + 1],
                    depth: 1,
                    parent: ROOT,
                    children: Vec::new(),
                    expanded: false,
                    alive: true,
                    anim_target: None,
                });
                kids.push(idx);
            }
        }
        dst[ROOT].children = kids;
        self.nodes = dst;
        self.view_lo = c_lo + self.view_lo * cw;
        self.view_hi = c_lo + self.view_hi * cw;
        if e.was_char {
            self.frame_chars -= 1;
        }
    }

    /// Forward frame advance: descend the root into the child the view has
    /// zoomed inside, re-anchoring coordinates in place (pure — the ledger
    /// records each level's exact partition for the way back). Triggered two
    /// ways: when the view is tiny relative to the frame (f64 precision), or
    /// when the crosshair is contained many levels deep (`soft_advance_depth`)
    /// so a confident model can't drive the frontier into the `max_depth`
    /// wall. A grouped character is absorbed as two levels (group box, then
    /// leaf).
    fn advance_frame_if_deep(&mut self) {
        loop {
            let deep_view = self.view_hi - self.view_lo < self.params.rebase_view_w;
            let contained = self.contained_path();
            let deep_tree = contained.len() >= self.params.soft_advance_depth;
            if !(deep_view || deep_tree) {
                return;
            }
            if !self.advance_frame_once() {
                return;
            }
        }
    }

    fn advance_frame_once(&mut self) -> bool {
        if !self.nodes[ROOT].expanded {
            return false;
        }
        // The shallowest root child the whole view sits inside.
        let opens = self.nodes[ROOT].opens;
        let Some((k, &child)) = self.nodes[ROOT]
            .children
            .iter()
            .enumerate()
            .find(|(_, &c)| {
                let n = &self.nodes[c];
                n.alive && n.lo <= self.view_lo && self.view_hi <= n.hi
            })
        else {
            return false; // view straddles a boundary; nothing to descend into
        };
        let (c_lo, c_hi) = (self.nodes[child].lo, self.nodes[child].hi);
        let is_leaf = self.nodes[child].as_char().is_some();
        if let Some(sym) = self.nodes[child].as_char() {
            if self.frame_chars < self.committed.len() {
                // Mid-retraction the view may still sit in the last committed
                // child while off the newer path; only absorb the committed
                // char, never a divergent one.
                if self.committed[self.frame_chars] != sym {
                    return false;
                }
            } else {
                // The view has zoomed entirely inside a not-yet-committed leaf
                // (fast steering outran the containment debounce). Commit it
                // now so the frame can advance and depth stays bounded; a
                // back-zoom still retracts it through the ledger.
                self.lm.learn(&self.committed, sym);
                self.committed.push(sym);
                self.events
                    .push(TextEvent::Commit(self.alphabet.char_of(sym).to_string()));
                self.commit_candidate = None;
                self.commit_timer = 0.0;
            }
        }
        let mut cuts: Vec<f64> = self.nodes[ROOT]
            .children
            .iter()
            .map(|&i| self.nodes[i].lo)
            .collect();
        cuts.push(1.0);
        self.ledger.push(LedgerEntry {
            cuts: cuts.into(),
            opens,
            child_idx: k,
            was_char: is_leaf,
        });
        let scale = 1.0 / (c_hi - c_lo);
        let offset = -c_lo * scale;
        let mut dst = Vec::with_capacity(self.nodes.len());
        copy_subtree(&self.nodes, child, &mut dst, ROOT, 0, offset, scale);
        dst[ROOT].lo = 0.0;
        dst[ROOT].hi = 1.0;
        self.nodes = dst;
        self.view_lo = offset + self.view_lo * scale;
        self.view_hi = offset + self.view_hi * scale;
        if is_leaf {
            self.frame_chars += 1;
        }
        true
    }

    /// Lazy, screen-driven expansion and eviction (SPEC §5.1). Thresholds are
    /// pixels: screen size *is* cumulative probability × zoom.
    fn maintain_tree(&mut self) {
        if self.nodes.len() > self.params.max_nodes {
            self.compact();
        }
        let w = self.view_hi - self.view_lo;
        // No point materializing deeper than where the frame will advance:
        // once the crosshair is `soft_advance_depth` in, the root descends and
        // depth resets, so a small lookahead beyond it is all that's ever
        // seen. Keeping the frontier here (well under `max_depth`) means the
        // depth wall can never strand a *reachable* box without children.
        let expand_cap = (self.params.soft_advance_depth + 8).min(self.params.max_depth);
        let mut stack = vec![ROOT];
        while let Some(i) = stack.pop() {
            let n = &self.nodes[i];
            if !n.alive {
                continue;
            }
            let (lo, hi, depth, expanded) = (n.lo, n.hi, n.depth, n.expanded);
            let visible = hi > self.view_lo && lo < self.view_hi;
            let h_px = (hi - lo) / w * self.viewport_px;
            if expanded {
                if depth > 0 && (!visible || h_px < self.params.evict_px) {
                    self.evict_children(i);
                } else {
                    stack.extend(self.nodes[i].children.iter().copied());
                }
            } else if visible
                && h_px > self.params.expand_px
                && depth < expand_cap
                && self.nodes.len() < self.params.max_nodes
            {
                self.expand(i);
                stack.extend(self.nodes[i].children.iter().copied());
            }
        }
    }

    /// Drop dead arena slots. Geometry is untouched — no visual effect.
    fn compact(&mut self) {
        let mut dst = Vec::with_capacity(self.nodes.len() / 2);
        copy_subtree(&self.nodes, ROOT, &mut dst, ROOT, 0, 0.0, 1.0);
        self.nodes = dst;
    }

    fn expand(&mut self, i: usize) {
        let (lo, hi, depth, opens) = {
            let n = &self.nodes[i];
            (n.lo, n.hi, n.depth, n.opens)
        };
        let ctx = self.context_of(i);
        let dist = self.lm.predict(&ctx);
        debug_assert_eq!(dist.len(), self.alphabet.len());
        let items: Vec<Item> = self.alphabet.items(opens).to_vec();
        let weights = self.alphabet.item_weights(opens, &dist);
        let total: f64 = weights.iter().sum::<f64>().max(f64::MIN_POSITIVE);
        let w = hi - lo;
        let mut cursor = lo;
        let mut children = Vec::with_capacity(items.len());
        for (k, &item) in items.iter().enumerate() {
            let c_lo = cursor;
            // Last child absorbs rounding so children exactly partition [lo,hi).
            let c_hi = if k + 1 == items.len() {
                hi
            } else {
                cursor + weights[k] / total * w
            };
            cursor = c_hi;
            let (label, c_opens) = label_and_opens(&self.alphabet, item);
            children.push(self.nodes.len());
            self.nodes.push(Node {
                label,
                opens: c_opens,
                lo: c_lo,
                hi: c_hi,
                depth: depth + 1,
                parent: i,
                children: Vec::new(),
                expanded: false,
                alive: true,
                anim_target: None,
            });
        }
        let n = &mut self.nodes[i];
        n.children = children;
        n.expanded = true;
    }

    fn evict_children(&mut self, i: usize) {
        // Cancel LM work for every expanded node about to disappear (§6.1).
        // Contexts must be computed while parent links are still intact.
        // A group box adds no character, so it shares its context with the
        // nearest non-group ancestor — which survives this eviction still
        // expanded (it must be, for `i` to exist). Releasing that shared
        // context would cancel an upgrade the ancestor is still waiting on,
        // stranding it on fallback sizing; skip it and any dying descendants
        // that share it (the group-only chain under `i`).
        let shared: Option<Vec<u8>> = self.nodes[i].is_group().then(|| self.context_of(i));
        let mut expanded_ctxs = match &shared {
            Some(_) => Vec::new(),
            None => vec![self.context_of(i)],
        };
        let mut walk = self.nodes[i].children.clone();
        while let Some(c) = walk.pop() {
            if self.nodes[c].alive && self.nodes[c].expanded {
                let ctx = self.context_of(c);
                if shared.as_ref() != Some(&ctx) {
                    expanded_ctxs.push(ctx);
                }
            }
            walk.extend(self.nodes[c].children.iter().copied());
        }
        for ctx in expanded_ctxs {
            self.lm.release(&ctx);
        }
        let mut stack = std::mem::take(&mut self.nodes[i].children);
        self.nodes[i].expanded = false;
        self.nodes[i].anim_target = None;
        while let Some(c) = stack.pop() {
            self.nodes[c].alive = false;
            self.nodes[c].anim_target = None;
            stack.extend(std::mem::take(&mut self.nodes[c].children));
        }
    }

    /// Frame-absorbed text plus the leaf characters from frame root to node
    /// `i` (group boxes on the path contribute no character).
    fn context_of(&self, i: usize) -> Vec<u8> {
        let mut path = Vec::new();
        let mut cur = i;
        while cur != ROOT {
            if let Some(s) = self.nodes[cur].as_char() {
                path.push(s);
            }
            cur = self.nodes[cur].parent;
        }
        let mut ctx = self.committed[..self.frame_chars].to_vec();
        ctx.extend(path.iter().rev());
        ctx
    }

    /// Chain of nodes (root-child first) that fully contain the view, including
    /// group boxes.
    fn contained_path(&self) -> Vec<usize> {
        let mut out = Vec::new();
        let mut cur = ROOT;
        'outer: loop {
            for &c in &self.nodes[cur].children {
                let n = &self.nodes[c];
                if n.alive && n.lo <= self.view_lo && n.hi >= self.view_hi {
                    out.push(c);
                    cur = c;
                    continue 'outer;
                }
            }
            break;
        }
        out
    }

    /// Chain of nodes containing the view center — the crosshair path: what
    /// will be written if the user keeps steering here (group boxes included).
    fn center_path(&self) -> Vec<usize> {
        let c = (self.view_lo + self.view_hi) / 2.0;
        let mut out = Vec::new();
        let mut cur = ROOT;
        'outer: loop {
            for &ch in &self.nodes[cur].children {
                let n = &self.nodes[ch];
                if n.alive && n.lo <= c && c < n.hi {
                    out.push(ch);
                    cur = ch;
                    continue 'outer;
                }
            }
            break;
        }
        out
    }

    /// Leaf-character symbols under the crosshair, for the ribbon preview.
    /// Empty while the crosshair path disagrees with the committed tail
    /// (i.e. mid-retraction).
    pub fn pending_syms(&self) -> Vec<u8> {
        let tail = &self.committed[self.frame_chars..];
        let syms: Vec<u8> = self
            .center_path()
            .iter()
            .filter_map(|&i| self.nodes[i].as_char())
            .collect();
        if syms.len() < tail.len() || !syms.starts_with(tail) {
            return Vec::new();
        }
        syms[tail.len()..].to_vec()
    }

    pub fn pending_text(&self) -> String {
        self.alphabet.decode(&self.pending_syms())
    }

    /// Does the deepest committed node still hold the view, within the
    /// retraction hysteresis? A missing tree link counts as "no".
    fn committed_node_holds_view(&self) -> bool {
        let eps = self.params.retract_eps * (self.view_hi - self.view_lo);
        let mut cur = ROOT;
        for &s in &self.committed[self.frame_chars..] {
            match self.descend_one_char(cur, s) {
                Some(x) => cur = x,
                None => return false,
            }
        }
        let n = &self.nodes[cur];
        n.lo - eps <= self.view_lo && self.view_hi <= n.hi + eps
    }

    /// Debounced text watermark, both directions (SPEC §5.2/§8.3): a node
    /// must contain the view for the debounce period to commit; the view must
    /// stay outside the deepest committed node for the same period to retract
    /// one symbol. Geometry never changes here — only the buffer and the LM.
    fn update_commit(&mut self, dt: f64, engaged: bool) {
        let contained = self.contained_path();
        // Only leaf characters commit; group boxes on the path are transparent.
        let chars: Vec<usize> = contained
            .iter()
            .copied()
            .filter(|&i| self.nodes[i].as_char().is_some())
            .collect();
        let tail = self.tail_len();
        let on_track = chars.len() >= tail
            && chars[..tail]
                .iter()
                .map(|&i| self.nodes[i].as_char().expect("filtered to chars"))
                .eq(self.committed[self.frame_chars..].iter().copied());
        if on_track && chars.len() > tail {
            self.outside_timer = 0.0;
            let cand = self.nodes[chars[tail]].as_char().expect("char");
            if self.commit_candidate != Some(cand) {
                self.commit_candidate = Some(cand);
                self.commit_timer = 0.0;
                return;
            }
            if engaged {
                self.commit_timer += dt;
            }
            if self.commit_timer < self.params.commit_debounce {
                return;
            }
            // Commit the whole contained chain — deeper containment implies
            // the first link was contained throughout the debounce.
            let mut text = String::new();
            for &id in &chars[tail..] {
                let sym = self.nodes[id].as_char().expect("char");
                self.lm.learn(&self.committed, sym);
                self.committed.push(sym);
                text.push(self.alphabet.char_of(sym));
            }
            self.events.push(TextEvent::Commit(text));
            self.commit_candidate = None;
            self.commit_timer = 0.0;
        } else if on_track {
            self.commit_candidate = None;
            self.commit_timer = 0.0;
            self.outside_timer = 0.0;
        } else {
            self.commit_candidate = None;
            self.commit_timer = 0.0;
            if self.committed_node_holds_view() {
                self.outside_timer = 0.0;
                return;
            }
            if engaged {
                self.outside_timer += dt;
            }
            if self.outside_timer < self.params.commit_debounce {
                return;
            }
            self.outside_timer = 0.0;
            let sym = self.committed.pop().expect("tail > 0 when off track");
            self.lm.unlearn(&self.committed, sym);
            self.events
                .push(TextEvent::Retract(self.alphabet.char_of(sym).to_string()));
        }
    }

    /// On release, commit the crosshair-path characters that dominate the view
    /// (height ≥ `commit_release_frac`), ahead of the containment debounce, so
    /// letting go lands the letter you were steering at. Geometry is untouched,
    /// exactly like a debounced commit — only the watermark and the LM move.
    /// A no-op while the crosshair path disagrees with the committed tail
    /// (mid-retraction) or nothing yet dominates.
    fn commit_under_crosshair(&mut self) {
        let w = self.view_hi - self.view_lo;
        let chars: Vec<usize> = self
            .center_path()
            .iter()
            .copied()
            .filter(|&i| self.nodes[i].as_char().is_some())
            .collect();
        let tail = self.tail_len();
        if chars.len() <= tail {
            return;
        }
        let on_track = chars[..tail]
            .iter()
            .map(|&i| self.nodes[i].as_char().expect("char"))
            .eq(self.committed[self.frame_chars..].iter().copied());
        if !on_track {
            return;
        }
        let mut text = String::new();
        for &id in &chars[tail..] {
            let (lo, hi, sym) = {
                let n = &self.nodes[id];
                (n.lo, n.hi, n.as_char().expect("char"))
            };
            if hi - lo < self.params.commit_release_frac * w {
                break; // deeper boxes are smaller still; stop at the first misfit
            }
            self.lm.learn(&self.committed, sym);
            self.committed.push(sym);
            text.push(self.alphabet.char_of(sym));
        }
        if !text.is_empty() {
            self.events.push(TextEvent::Commit(text));
            self.commit_candidate = None;
            self.commit_timer = 0.0;
        }
    }

    /// Interval, in current-frame coordinates, of a continuation `next`
    /// (leaf characters) beyond the committed text. Follows actual tree
    /// geometry where nodes exist (the LM may have learned since a node was
    /// laid out) and extends with fresh LM predictions below the tree,
    /// descending through group boxes as needed. Used by the oracle sim.
    pub fn path_interval(&mut self, next: &[u8]) -> (f64, f64) {
        let mut syms: Vec<u8> = self.committed[self.frame_chars..].to_vec();
        syms.extend_from_slice(next);
        let mut ctx: Vec<u8> = self.committed[..self.frame_chars].to_vec();
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        let mut node = Some(ROOT);
        let mut opens = self.nodes[ROOT].opens;
        for &c in &syms {
            // One character = descend the layout from `opens` down to its leaf,
            // possibly crossing a group level.
            loop {
                // The frame can be rooted inside a group box (the view zoomed
                // into it); a non-member character then isn't reachable without
                // first backing out. Report the whole current interval so the
                // oracle steers outward rather than the helper panicking.
                let Some((k, is_leaf)) = self.alphabet.item_leading_to(opens, c) else {
                    return (lo, hi);
                };
                match node {
                    Some(i) if self.nodes[i].expanded => {
                        let ch = self.nodes[i].children[k];
                        lo = self.nodes[ch].lo;
                        hi = self.nodes[ch].hi;
                        node = Some(ch);
                        opens = self.nodes[ch].opens;
                    }
                    _ => {
                        node = None;
                        let dist = self.lm.predict(&ctx);
                        let weights = self.alphabet.item_weights(opens, &dist);
                        let total: f64 = weights.iter().sum::<f64>().max(f64::MIN_POSITIVE);
                        let before: f64 = weights[..k].iter().sum();
                        let w = hi - lo;
                        let n_lo = lo + w * before / total;
                        hi = n_lo + w * weights[k] / total;
                        lo = n_lo;
                        opens = self.alphabet.opens_of(self.alphabet.items(opens)[k]);
                    }
                }
                if is_leaf {
                    break;
                }
            }
            ctx.push(c);
            // Next character always starts from the root layout.
            opens = self.alphabet.root_layout();
        }
        (lo, hi)
    }

    /// Boxes to draw, parents before children.
    pub fn visible(&self) -> Vec<BoxView> {
        let w = self.view_hi - self.view_lo;
        let path = self.center_path();
        let frame_depth = self.ledger.len();
        let mut out: Vec<BoxView> = self
            .nodes
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, n)| n.alive && n.hi > self.view_lo && n.lo < self.view_hi)
            .map(|(i, n)| {
                let (sym, label, group) = match n.label {
                    NodeLabel::Char(s) => (s, self.alphabet.label(s), None),
                    NodeLabel::Group(g) => (
                        0,
                        self.alphabet.group_label(g),
                        Some(self.alphabet.group_color(g)),
                    ),
                };
                BoxView {
                    sym,
                    label,
                    group,
                    id: i,
                    parent: n.parent,
                    depth: frame_depth + n.depth,
                    y0: (n.lo - self.view_lo) / w,
                    y1: (n.hi - self.view_lo) / w,
                    on_path: path.contains(&i),
                }
            })
            .collect();
        out.sort_by(|a, b| a.depth.cmp(&b.depth).then(a.y0.total_cmp(&b.y0)));
        out
    }
}

/// A single fresh root node covering the whole frame [0, 1), opening into the
/// root layout. The tree starts here, and `Engine::reset` returns to this.
fn root_node(root_layout: u32) -> Node {
    Node {
        label: NodeLabel::Char(0),
        opens: root_layout,
        lo: 0.0,
        hi: 1.0,
        depth: 0,
        parent: ROOT,
        children: Vec::new(),
        expanded: false,
        alive: true,
        anim_target: None,
    }
}

/// A child node's label and the layout it opens into, for a layout item.
fn label_and_opens(alphabet: &Alphabet, item: Item) -> (NodeLabel, u32) {
    match item {
        Item::Leaf(s) => (NodeLabel::Char(s), alphabet.root_layout()),
        Item::Group(g) => (NodeLabel::Group(g), g),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::UniformModel;

    fn engine() -> Engine {
        let a = Alphabet::english_v1();
        let n = a.len();
        Engine::new(a, Box::new(UniformModel(n)), EngineParams::default())
    }

    /// Steer toward `target` (symbols) using the same aim the oracle uses,
    /// without noise/delay. Returns false on timeout.
    fn steer_to(e: &mut Engine, target: &[u8], max_frames: usize) -> bool {
        for _ in 0..max_frames {
            if e.committed_syms() == target {
                return true;
            }
            let committed = e.committed_syms().to_vec();
            let input = if !target.starts_with(&committed) {
                PointerInput {
                    dx: -1.0,
                    y: 0.5,
                    engaged: true,
                }
            } else {
                let rest = &target[committed.len()..];
                let depth = rest.len().min(4);
                let (lo, hi) = e.path_interval(&rest[..depth]);
                let (v_lo, v_hi) = e.view();
                let y = ((lo + hi) / 2.0 - v_lo) / (v_hi - v_lo);
                if (0.0..=1.0).contains(&y) {
                    PointerInput {
                        dx: 1.0,
                        y,
                        engaged: true,
                    }
                } else {
                    PointerInput {
                        dx: -0.3,
                        y: y.clamp(0.0, 1.0),
                        engaged: true,
                    }
                }
            };
            e.step(input, 1.0 / 60.0);
        }
        false
    }

    #[test]
    fn children_partition_parent() {
        let mut e = engine();
        e.set_viewport_px(1000.0);
        e.step(
            PointerInput {
                dx: 0.0,
                y: 0.5,
                engaged: false,
            },
            1.0 / 60.0,
        );
        let root = &e.nodes[ROOT];
        assert!(root.expanded);
        let mut cursor = 0.0;
        for &c in &root.children {
            let n = &e.nodes[c];
            assert!((n.lo - cursor).abs() < 1e-12);
            cursor = n.hi;
        }
        assert!((cursor - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zoom_in_commits_and_retract_restores() {
        let mut e = engine();
        e.set_viewport_px(1000.0);
        e.speed_bits = 8.0;
        assert!(steer_to(&mut e, &[0], 600), "should commit 'a'");
        // Now zoom straight out; the commit must retract.
        for _ in 0..600 {
            e.step(
                PointerInput {
                    dx: -1.0,
                    y: 0.5,
                    engaged: true,
                },
                1.0 / 60.0,
            );
            if e.committed_syms().is_empty() {
                break;
            }
        }
        assert!(
            e.committed_syms().is_empty(),
            "retraction should undo commit"
        );
    }

    #[test]
    fn text_events_mirror_commits_and_retractions() {
        let mut e = engine();
        e.set_viewport_px(1000.0);
        e.speed_bits = 8.0;
        assert!(steer_to(&mut e, &[0], 600), "should commit 'a'");
        let evs = e.drain_text_events();
        assert!(
            evs.iter()
                .any(|ev| matches!(ev, TextEvent::Commit(s) if s.contains('a'))),
            "commit must emit an event: {evs:?}"
        );
        for _ in 0..600 {
            e.step(
                PointerInput {
                    dx: -1.0,
                    y: 0.5,
                    engaged: true,
                },
                1.0 / 60.0,
            );
            if e.committed_syms().is_empty() {
                break;
            }
        }
        let evs = e.drain_text_events();
        assert!(
            evs.iter()
                .any(|ev| matches!(ev, TextEvent::Retract(s) if s == "a")),
            "retraction must emit an event: {evs:?}"
        );
        assert!(
            e.drain_text_events().is_empty(),
            "drain must clear the queue"
        );
    }

    #[test]
    fn deterministic_replay() {
        let run = || {
            let mut e = engine();
            e.set_viewport_px(900.0);
            e.speed_bits = 6.0;
            let mut trace = Vec::new();
            let mut y = 0.3;
            for i in 0..2000 {
                y = 0.5 + 0.45 * ((i as f64) * 0.013).sin();
                e.step(
                    PointerInput {
                        dx: 0.9,
                        y,
                        engaged: true,
                    },
                    1.0 / 60.0,
                );
                trace.push(e.view());
            }
            let _ = y;
            (e.committed_text(), trace.last().copied())
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn view_stays_sane_at_document_root() {
        let mut e = engine();
        e.set_viewport_px(800.0);
        for _ in 0..1000 {
            e.step(
                PointerInput {
                    dx: -1.0,
                    y: 0.1,
                    engaged: true,
                },
                1.0 / 60.0,
            );
        }
        let (lo, hi) = e.view();
        assert!(lo >= -1e-9 && hi <= 1.0 + 1e-9 && hi > lo);
    }

    #[test]
    fn pan_scrolls_without_zoom() {
        let mut e = engine();
        e.set_viewport_px(800.0);
        // Zoom in around the center so there is room to pan.
        for _ in 0..120 {
            e.step(
                PointerInput {
                    dx: 0.8,
                    y: 0.5,
                    engaged: true,
                },
                1.0 / 60.0,
            );
        }
        let (lo0, hi0) = e.view();
        let (c0, w0) = ((lo0 + hi0) / 2.0, hi0 - lo0);
        for _ in 0..60 {
            e.step(
                PointerInput {
                    dx: 0.0,
                    y: 0.1,
                    engaged: true,
                },
                1.0 / 60.0,
            );
        }
        let (lo1, hi1) = e.view();
        let c1 = (lo1 + hi1) / 2.0;
        assert!(
            c1 < c0 - 0.1 * w0,
            "pointing up must scroll up: {c0} -> {c1}"
        );
        assert!(
            ((hi1 - lo1) - w0).abs() < w0 * 1e-9,
            "dx=0 must not zoom: {w0} -> {}",
            hi1 - lo1
        );
    }

    #[test]
    fn frame_rebase_roundtrip_is_exact() {
        let mut e = engine();
        // Force a frame advance after almost every commit so the ledger and
        // climb-out paths are exercised hard.
        e.params.rebase_view_w = 0.5;
        e.set_viewport_px(1000.0);
        e.speed_bits = 8.0;
        // Capture the document root partition before anything moves.
        e.step(
            PointerInput {
                dx: 0.0,
                y: 0.5,
                engaged: false,
            },
            1.0 / 60.0,
        );
        let cuts0: Vec<f64> = e.nodes[ROOT]
            .children
            .iter()
            .map(|&c| e.nodes[c].lo)
            .collect();

        let target = [1u8, 0, 2]; // "bac"
        assert!(steer_to(&mut e, &target, 3000), "should write the target");
        assert!(
            !e.ledger.is_empty(),
            "low rebase threshold must advance the frame"
        );

        // Zoom all the way back out; text retracts and frames climb.
        for _ in 0..3000 {
            e.step(
                PointerInput {
                    dx: -1.0,
                    y: 0.5,
                    engaged: true,
                },
                1.0 / 60.0,
            );
            if e.committed_syms().is_empty() && e.ledger.is_empty() {
                break;
            }
        }
        assert!(e.committed_syms().is_empty(), "full retraction");
        assert!(e.ledger.is_empty(), "frame back at document root");
        let cuts1: Vec<f64> = e.nodes[ROOT]
            .children
            .iter()
            .map(|&c| e.nodes[c].lo)
            .collect();
        assert_eq!(cuts0, cuts1, "climb-out must restore geometry exactly");
    }

    /// Async double: answers uniform immediately, then upgrades the root
    /// context to a peaked distribution on the second poll (the first poll of
    /// a step precedes expansion, mirroring a real in-flight worker).
    struct UpgradingModel {
        n: usize,
        polls: usize,
        upgrade: Option<Vec<f64>>,
    }

    impl LanguageModel for UpgradingModel {
        fn predict(&mut self, _context: &[u8]) -> Vec<f64> {
            vec![1.0 / self.n as f64; self.n]
        }

        fn poll_upgrades(&mut self) -> Vec<(Vec<u8>, Vec<f64>)> {
            self.polls += 1;
            if self.polls < 2 {
                return Vec::new();
            }
            self.upgrade
                .take()
                .map(|d| (Vec::new(), d))
                .into_iter()
                .collect()
        }

        fn alphabet_len(&self) -> usize {
            self.n
        }
    }

    #[test]
    fn upgrade_animates_and_preserves_partition() {
        let a = Alphabet::english_v1();
        let n = a.len();
        // Upgrade: symbol 0 gets half the mass, rest uniform.
        let mut target = vec![0.5 / (n - 1) as f64; n];
        target[0] = 0.5;
        let lm = UpgradingModel {
            n,
            polls: 0,
            upgrade: Some(target.clone()),
        };
        let mut e = Engine::new(a, Box::new(lm), EngineParams::default());
        e.set_viewport_px(1000.0);
        // Step 1 expands the root with uniform sizing; step 2 receives the
        // upgrade and starts animating.
        e.step(
            PointerInput {
                dx: 0.0,
                y: 0.5,
                engaged: false,
            },
            1.0 / 60.0,
        );
        e.step(
            PointerInput {
                dx: 0.0,
                y: 0.5,
                engaged: false,
            },
            1.0 / 60.0,
        );
        let w_uniform = 1.0 / n as f64;
        let first = e.nodes[ROOT].children[0];
        let w0 = e.nodes[first].hi - e.nodes[first].lo;
        assert!(
            w0 > w_uniform,
            "child 0 must start growing immediately: {w0}"
        );
        assert!(w0 < 0.5, "growth must be gradual, not a pop: {w0}");
        // Partition invariant must hold on every animation frame.
        for _ in 0..30 {
            e.step(
                PointerInput {
                    dx: 0.0,
                    y: 0.5,
                    engaged: false,
                },
                1.0 / 60.0,
            );
            let mut cursor = 0.0;
            for &c in &e.nodes[ROOT].children {
                assert!((e.nodes[c].lo - cursor).abs() < 1e-12);
                cursor = e.nodes[c].hi;
            }
            assert!((cursor - 1.0).abs() < 1e-12);
        }
        // ~0.5 s later the animation must have converged to the target.
        let w1 = e.nodes[first].hi - e.nodes[first].lo;
        assert!(
            (w1 - 0.5).abs() < 1e-6,
            "must converge to upgraded size: {w1}"
        );
        assert!(e.nodes[ROOT].anim_target.is_none(), "animation must finish");
    }

    #[test]
    fn release_commits_dominant_crosshair_letter() {
        let mut e = engine();
        e.set_viewport_px(1000.0);
        e.speed_bits = 6.0;
        let mut released = false;
        for _ in 0..600 {
            // Aim at the center of symbol 0's interval and zoom in.
            let (lo, hi) = e.path_interval(&[0]);
            let target = (lo + hi) / 2.0;
            let (v_lo, v_hi) = e.view();
            let w = v_hi - v_lo;
            let y = (target - v_lo) / w;
            // Has 'a' grown to dominate the view without fully engulfing it
            // (so the containment debounce hasn't fired)?
            let dominant = e.center_path().first().is_some_and(|&i| {
                let n = &e.nodes[i];
                n.as_char() == Some(0)
                    && n.hi - n.lo >= e.params.commit_release_frac * w
                    && !(n.lo <= v_lo && n.hi >= v_hi)
            });
            assert!(
                e.committed_syms().is_empty(),
                "must not auto-commit before dominance"
            );
            if dominant {
                // Let go: the dominant letter must commit even though it never
                // fully engulfed the view.
                e.step(
                    PointerInput {
                        dx: 0.0,
                        y,
                        engaged: false,
                    },
                    1.0 / 60.0,
                );
                assert_eq!(
                    e.committed_syms(),
                    [0],
                    "release must land the dominant letter"
                );
                let evs = e.drain_text_events();
                assert!(
                    evs.iter()
                        .any(|ev| matches!(ev, TextEvent::Commit(s) if s == "a")),
                    "release commit must emit an event: {evs:?}"
                );
                released = true;
                break;
            }
            e.step(
                PointerInput {
                    dx: 1.0,
                    y,
                    engaged: true,
                },
                1.0 / 60.0,
            );
        }
        assert!(
            released,
            "'a' should have become dominant under the crosshair"
        );
    }

    #[test]
    fn release_without_dominant_box_commits_nothing() {
        let mut e = engine();
        e.set_viewport_px(1000.0);
        // Expand the root but never zoom: no box dominates the full-width view.
        e.step(
            PointerInput {
                dx: 0.0,
                y: 0.5,
                engaged: true,
            },
            1.0 / 60.0,
        );
        e.step(
            PointerInput {
                dx: 0.0,
                y: 0.5,
                engaged: false,
            },
            1.0 / 60.0,
        );
        assert!(
            e.committed_syms().is_empty(),
            "a plain pause must not commit"
        );
    }

    #[test]
    fn pending_preview_appears_before_commit() {
        let mut e = engine();
        e.set_viewport_px(1000.0);
        e.speed_bits = 6.0;
        let (lo, hi) = e.path_interval(&[0]);
        let target = (lo + hi) / 2.0;
        let mut saw_preview = false;
        for _ in 0..600 {
            let (v_lo, v_hi) = e.view();
            let y = (target - v_lo) / (v_hi - v_lo);
            e.step(
                PointerInput {
                    dx: 1.0,
                    y,
                    engaged: true,
                },
                1.0 / 60.0,
            );
            if e.committed_syms().is_empty() && e.pending_syms().first() == Some(&0) {
                saw_preview = true;
            }
            if !e.committed_syms().is_empty() {
                break;
            }
        }
        assert!(saw_preview, "pending path must show 'a' before it commits");
    }

    /// Records every learn/unlearn as a signed count per (context, sym) in a
    /// shared map, so a balanced session leaves all counts at zero. Predictions
    /// stay uniform.
    struct CountingModel {
        n: usize,
        counts: std::sync::Arc<std::sync::Mutex<std::collections::HashMap<(Vec<u8>, u8), i64>>>,
    }

    impl LanguageModel for CountingModel {
        fn predict(&mut self, _context: &[u8]) -> Vec<f64> {
            vec![1.0 / self.n as f64; self.n]
        }
        fn learn(&mut self, context: &[u8], sym: u8) {
            *self
                .counts
                .lock()
                .unwrap()
                .entry((context.to_vec(), sym))
                .or_default() += 1;
        }
        fn unlearn(&mut self, context: &[u8], sym: u8) {
            *self
                .counts
                .lock()
                .unwrap()
                .entry((context.to_vec(), sym))
                .or_default() -= 1;
        }
        fn alphabet_len(&self) -> usize {
            self.n
        }
    }

    #[test]
    fn reset_clears_state_and_rewinds_lm() {
        let a = Alphabet::english_v1();
        let n = a.len();
        let counts = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
        let lm = CountingModel {
            n,
            counts: counts.clone(),
        };
        let mut e = Engine::new(a, Box::new(lm), EngineParams::default());
        e.set_viewport_px(1000.0);
        e.speed_bits = 8.0;
        // Low rebase threshold forces frame advances so the ledger is exercised.
        e.params.rebase_view_w = 0.5;
        assert!(steer_to(&mut e, &[1, 0, 2], 3000), "should write 'bac'");
        assert!(!e.committed_syms().is_empty());
        assert!(!e.ledger.is_empty(), "frame should have advanced");
        assert!(
            counts.lock().unwrap().values().any(|&c| c != 0),
            "committing must have taught the LM something to rewind"
        );
        e.drain_text_events();

        e.reset();

        assert!(e.committed_syms().is_empty(), "committed buffer cleared");
        assert!(e.ledger.is_empty(), "ledger cleared");
        assert_eq!(e.frame_chars, 0, "frame char count cleared");
        assert_eq!(e.view(), (0.0, 1.0), "view back at the root frame");
        assert_eq!(e.nodes.len(), 1, "tree back to a single root");
        assert_eq!(e.speed_bits, e.params.max_bits_per_sec, "speed restored");
        assert!(
            e.drain_text_events().is_empty(),
            "reset emits no text events"
        );
        assert!(
            counts.lock().unwrap().values().all(|&c| c == 0),
            "reset must unlearn every commit, leaving the LM at baseline"
        );
    }

    fn full_engine() -> Engine {
        let a = Alphabet::english_full();
        let n = a.len();
        Engine::new(a, Box::new(UniformModel(n)), EngineParams::default())
    }

    /// Sharply peaked deterministic Markov chain over the lowercase letters:
    /// the argmax next letter is a fixed function of the previous one, so the
    /// dominant boxes form one long followable chain — a "hold zoom into the
    /// biggest box" path the oracle never drives, and the worst case for
    /// runaway speculative depth.
    struct PeakedModel {
        n: usize,
    }
    impl PeakedModel {
        fn next_letter(prev: u8) -> u8 {
            (prev.wrapping_mul(7).wrapping_add(3)) % 26
        }
    }
    impl LanguageModel for PeakedModel {
        fn predict(&mut self, context: &[u8]) -> Vec<f64> {
            let n = self.n;
            let mut p = vec![0.01 / n as f64; n];
            let prev = context.last().copied().unwrap_or(3);
            p[Self::next_letter(prev) as usize] = 0.99;
            let s: f64 = p.iter().sum();
            for v in p.iter_mut() {
                *v /= s;
            }
            p
        }
        fn alphabet_len(&self) -> usize {
            self.n
        }
    }

    /// A human who zooms straight into the dominant box must never end up
    /// *inside* a box with no expandable children ("no boxes shown, typing
    /// impossible"). Uses a degenerately-confident model so the speculative
    /// frontier is driven as hard as possible; the box the crosshair is
    /// actually contained in must always be able to reveal its next level.
    #[test]
    fn deep_zoom_into_peaked_chain_never_wedges() {
        let a = Alphabet::english_full();
        let n = a.len();
        let mut e = Engine::new(a, Box::new(PeakedModel { n }), EngineParams::default());
        e.set_viewport_px(900.0);
        e.speed_bits = 6.0;
        let mut max_depth = 0usize;
        let mut max_nodes = 0usize;
        let mut wrote = 0usize;
        for frame in 0..6000 {
            let committed = e.committed_syms().to_vec();
            wrote = wrote.max(committed.len());
            let prev = committed.last().copied().unwrap_or(3);
            let arg = PeakedModel::next_letter(prev);
            let (lo, hi) = e.path_interval(&[arg]);
            let (v_lo, v_hi) = e.view();
            let y = ((lo + hi) / 2.0 - v_lo) / (v_hi - v_lo);
            e.step(
                PointerInput {
                    dx: 1.0,
                    y: y.clamp(0.0, 1.0),
                    engaged: true,
                },
                1.0 / 60.0,
            );
            max_depth = max_depth.max(
                e.nodes
                    .iter()
                    .filter(|x| x.alive)
                    .map(|x| x.depth)
                    .max()
                    .unwrap_or(0),
            );
            max_nodes = max_nodes.max(e.nodes.len());
            // The deepest box that fully contains the view is the one the user
            // is inside. Once it dominates the view it must be expanded with
            // live children, or there is nowhere to steer next.
            let (v_lo, v_hi) = e.view();
            if let Some(&deep) = e.contained_path().last() {
                let node = &e.nodes[deep];
                let h_px = (node.hi - node.lo) / (v_hi - v_lo) * 900.0;
                let dominates = h_px > 3.0 * e.params.expand_px;
                let childless = !node.expanded || node.children.iter().all(|&c| !e.nodes[c].alive);
                if dominates && childless {
                    panic!(
                        "wedged at frame {frame}: contained box depth={} h_px={:.0} has no \
                         children (committed len={}, max_depth={max_depth}, max_nodes={max_nodes})",
                        node.depth,
                        h_px,
                        e.committed_syms().len()
                    );
                }
            }
        }
        // And it must actually make progress, not just avoid wedging.
        assert!(wrote > 20, "should have written many chars, only {wrote}");
        eprintln!("no wedge: wrote={wrote} max_depth={max_depth} max_nodes={max_nodes}");
    }

    #[test]
    fn types_the_man_through_capitals() {
        let mut e = full_engine();
        e.set_viewport_px(1000.0);
        e.speed_bits = 8.0;
        let target = e.alphabet.encode("The man");
        eprintln!("target syms = {target:?}");
        let ok = steer_to(&mut e, &target, 30000);
        eprintln!("committed = {:?}", e.committed_text());
        assert!(ok, "should type 'The man', got {:?}", e.committed_text());
    }

    #[test]
    fn types_capital_phrase_with_forced_frame_advance() {
        let mut e = full_engine();
        // Advance the frame after nearly every commit so group-level ledger
        // absorption and climb-out are exercised hard on a capital path.
        e.params.rebase_view_w = 0.5;
        e.set_viewport_px(1000.0);
        e.speed_bits = 8.0;
        let target = e.alphabet.encode("The man");
        let ok = steer_to(&mut e, &target, 40000);
        eprintln!(
            "committed = {:?} ledger={}",
            e.committed_text(),
            e.ledger.len()
        );
        assert!(
            ok,
            "should type 'The man' under forced rebase, got {:?}",
            e.committed_text()
        );
    }

    /// A grouped character ('A', inside the capitals group) is reachable and
    /// commits the right leaf, exercising the group-box tree level.
    #[test]
    fn writes_a_grouped_capital() {
        let mut e = full_engine();
        e.set_viewport_px(1000.0);
        e.speed_bits = 8.0;
        let cap_a = e.alphabet.index_of('A').unwrap();
        assert!(steer_to(&mut e, &[cap_a], 4000), "should commit 'A'");
        assert_eq!(e.committed_text(), "A");
    }

    /// Grouping must not disturb leaf geometry: a lowercase target writes the
    /// same on the grouped alphabet as on the flat one (identical LM), and
    /// round-trips through frame advance/climb exactly.
    #[test]
    fn grouped_alphabet_roundtrips_lowercase() {
        let mut e = full_engine();
        e.params.rebase_view_w = 0.5;
        e.set_viewport_px(1000.0);
        e.speed_bits = 8.0;
        let target = e.alphabet.encode("cab");
        assert!(steer_to(&mut e, &target, 5000), "should write 'cab'");
        assert!(!e.ledger.is_empty(), "frame should have advanced");
        for _ in 0..5000 {
            e.step(
                PointerInput {
                    dx: -1.0,
                    y: 0.5,
                    engaged: true,
                },
                1.0 / 60.0,
            );
            if e.committed_syms().is_empty() && e.ledger.is_empty() {
                break;
            }
        }
        assert!(e.committed_syms().is_empty(), "full retraction");
        assert!(e.ledger.is_empty(), "frame back at document root");
        assert_eq!(e.frame_chars, 0);
    }

    /// Backtracking to the document start must restore the tree's labels, not
    /// just its geometry. Regression: climb-out used to transplant the old
    /// frame root without restamping its label, so every spine node of the
    /// undone path (including the capitals group box) rendered as the
    /// placeholder symbol 0 ('a') and re-committed the wrong characters.
    #[test]
    fn climb_out_restores_labels_after_grouped_backtrack() {
        let mut e = full_engine();
        // Force frame advances so backtracking has many levels to climb out of.
        e.params.rebase_view_w = 0.5;
        e.set_viewport_px(1000.0);
        e.speed_bits = 8.0;
        let target = e.alphabet.encode("I have");
        assert!(steer_to(&mut e, &target, 40000), "should write 'I have'");
        assert!(!e.ledger.is_empty(), "frame should have advanced");
        for _ in 0..40000 {
            e.step(
                PointerInput {
                    dx: -1.0,
                    y: 0.5,
                    engaged: true,
                },
                1.0 / 60.0,
            );
            if e.committed_syms().is_empty() && e.ledger.is_empty() {
                break;
            }
        }
        assert!(e.committed_syms().is_empty(), "full retraction");
        assert!(e.ledger.is_empty(), "frame back at document root");
        // Every child's label/opens must match its slot in the parent layout.
        for i in 0..e.nodes.len() {
            if !e.nodes[i].alive || !e.nodes[i].expanded {
                continue;
            }
            let items = e.alphabet.items(e.nodes[i].opens).to_vec();
            for (k, &c) in e.nodes[i].children.iter().enumerate() {
                let (label, opens) = label_and_opens(&e.alphabet, items[k]);
                assert_eq!(e.nodes[c].opens, opens, "node {c} opens wrong layout");
                let ok = matches!(
                    (label, e.nodes[c].label),
                    (NodeLabel::Char(a), NodeLabel::Char(b)) if a == b
                ) || matches!(
                    (label, e.nodes[c].label),
                    (NodeLabel::Group(a), NodeLabel::Group(b)) if a == b
                );
                assert!(
                    ok,
                    "node {c} label {:?} does not match layout item {label:?}",
                    e.nodes[c].label
                );
            }
        }
        // The capitals group box is back where it belongs, not a phantom 'a'.
        let boxes = e.visible();
        assert!(
            boxes.iter().any(|b| b.group == Some(GroupColor::Capitals)),
            "capitals group box must be visible again at the root"
        );
        // Moving forward again replays the exact same text.
        assert!(steer_to(&mut e, &target, 40000), "retype after backtrack");
        assert_eq!(e.committed_text(), "I have");
    }

    /// Uniform predictions plus a log of every released context.
    struct ReleaseLog {
        n: usize,
        released: std::sync::Arc<std::sync::Mutex<Vec<Vec<u8>>>>,
    }

    impl LanguageModel for ReleaseLog {
        fn predict(&mut self, _context: &[u8]) -> Vec<f64> {
            vec![1.0 / self.n as f64; self.n]
        }
        fn release(&mut self, context: &[u8]) {
            self.released.lock().unwrap().push(context.to_vec());
        }
        fn alphabet_len(&self) -> usize {
            self.n
        }
    }

    /// Evicting a group box's children must not release the context it shares
    /// with its (still expanded) non-group ancestor. Regression: releasing it
    /// cancelled the ancestor's pending distribution upgrade, stranding the
    /// ancestor on fallback sizing for as long as it stayed on screen.
    #[test]
    fn evicting_group_children_keeps_shared_context() {
        let a = Alphabet::english_full();
        let n = a.len();
        let released = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let lm = ReleaseLog {
            n,
            released: released.clone(),
        };
        let mut e = Engine::new(a, Box::new(lm), EngineParams::default());
        e.set_viewport_px(1000.0);
        // Expand the root; the group boxes are large enough (uniform model)
        // to expand too. Root and groups all share the empty context.
        e.step(
            PointerInput {
                dx: 0.0,
                y: 0.5,
                engaged: false,
            },
            1.0 / 60.0,
        );
        let groups: Vec<usize> = e.nodes[ROOT]
            .children
            .iter()
            .copied()
            .filter(|&c| e.nodes[c].is_group())
            .collect();
        assert!(
            groups.iter().any(|&g| e.nodes[g].expanded),
            "at least one group box must have expanded"
        );
        // Park the view deep inside 'a': the groups go invisible and their
        // children are evicted while the root stays expanded.
        e.view_lo = 0.0001;
        e.view_hi = 0.0002;
        e.maintain_tree();
        assert!(
            groups.iter().all(|&g| !e.nodes[g].expanded),
            "groups off-screen must have their children evicted"
        );
        assert!(e.nodes[ROOT].expanded, "root must stay expanded");
        assert!(
            !released.lock().unwrap().iter().any(|c| c.is_empty()),
            "the root's context must not be released: {:?}",
            released.lock().unwrap()
        );
    }

    /// The visible set marks group boxes with a color and their label, and
    /// leaves as plain characters.
    #[test]
    fn visible_marks_group_boxes() {
        let mut e = full_engine();
        e.set_viewport_px(1000.0);
        e.step(
            PointerInput {
                dx: 0.0,
                y: 0.5,
                engaged: false,
            },
            1.0 / 60.0,
        );
        let boxes = e.visible();
        assert!(
            boxes.iter().any(|b| b.group == Some(GroupColor::Capitals)),
            "a capitals group box must be visible at the top level"
        );
        assert!(
            boxes.iter().any(|b| b.group.is_none() && b.label == 'a'),
            "plain letter boxes must still be present"
        );
    }

    /// Adaptive speed rises under smooth forward steering and is cut by a
    /// zoom reversal.
    #[test]
    fn adaptive_speed_rises_and_drops() {
        let a = Alphabet::english_v1();
        let n = a.len();
        let mut p = EngineParams {
            adaptive_speed: true,
            ..EngineParams::default()
        };
        p.min_bits_per_sec = 1.5;
        p.max_bits_per_sec = 8.0;
        let mut e = Engine::new(a, Box::new(UniformModel(n)), p);
        e.set_viewport_px(1000.0);
        let start = e.speed_bits;
        for _ in 0..300 {
            e.step(
                PointerInput {
                    dx: 1.0,
                    y: 0.5,
                    engaged: true,
                },
                1.0 / 60.0,
            );
        }
        let risen = e.speed_bits;
        assert!(
            risen > start + 0.5,
            "smooth steering must raise the cap: {start} -> {risen}"
        );
        // A hard reversal cuts it immediately.
        e.step(
            PointerInput {
                dx: 1.0,
                y: 0.5,
                engaged: true,
            },
            1.0 / 60.0,
        );
        e.step(
            PointerInput {
                dx: -1.0,
                y: 0.5,
                engaged: true,
            },
            1.0 / 60.0,
        );
        assert!(
            e.speed_bits < risen,
            "a reversal must cut the cap: {risen} -> {}",
            e.speed_bits
        );
        assert!(e.speed_bits >= e.params.min_bits_per_sec);
    }
}

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

use crate::alphabet::Alphabet;
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
    /// Expansion depth limit below the current frame root (safety net; the
    /// pixel thresholds are the real gate).
    pub max_depth: usize,
    /// Arena size that triggers compaction (dead-node garbage collection).
    pub max_nodes: usize,
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
            max_depth: 64,
            max_nodes: 4000,
        }
    }
}

struct Node {
    sym: u8,
    lo: f64,
    hi: f64,
    /// Depth below the current frame root (shifts when the frame moves).
    depth: usize,
    parent: usize,
    children: Vec<usize>,
    expanded: bool,
    alive: bool,
}

/// One frame-absorbed symbol: the parent frame's complete child partition
/// (`cuts[i]..cuts[i+1]` is child `i`) plus which child the frame descended
/// into. Climbing back out re-lays the parent exactly as it was on screen,
/// immune to LM drift.
#[derive(Clone, Debug)]
struct LedgerEntry {
    cuts: Box<[f64]>,
    sym: u8,
}

/// A renderable box: vertical extent as fractions of the canvas (may exceed
/// [0,1]; the renderer clips).
#[derive(Clone, Copy, Debug)]
pub struct BoxView {
    pub sym: u8,
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
    /// Full committed text. The first `ledger.len()` symbols are absorbed
    /// into the frame; the rest (the "tail") are nodes in the tree.
    committed: Vec<u8>,
    ledger: Vec<LedgerEntry>,
    commit_candidate: Option<u8>, // symbol of first uncommitted contained node
    commit_timer: f64,
    outside_timer: f64,
    viewport_px: f64,
    /// Current speed cap in bits/s; UI may adjust, sim sets directly.
    pub speed_bits: f64,
}

const ROOT: usize = 0;
/// View may exceed the deepest committed node by this fraction of its width
/// before a retraction is considered (overshoot hysteresis).
const RETRACT_EPS: f64 = 0.02;
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
        sym: n.sym,
        lo: offset + n.lo * scale,
        hi: offset + n.hi * scale,
        depth,
        parent,
        children: Vec::new(),
        expanded: n.expanded,
        alive: true,
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
        Engine {
            alphabet,
            lm,
            params,
            nodes: vec![Node {
                sym: 0,
                lo: 0.0,
                hi: 1.0,
                depth: 0,
                parent: ROOT,
                children: Vec::new(),
                expanded: false,
                alive: true,
            }],
            view_lo: 0.0,
            view_hi: 1.0,
            committed: Vec::new(),
            ledger: Vec::new(),
            commit_candidate: None,
            commit_timer: 0.0,
            outside_timer: 0.0,
            viewport_px: 800.0,
            speed_bits: params.max_bits_per_sec,
        }
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

    fn tail_len(&self) -> usize {
        self.committed.len() - self.ledger.len()
    }

    /// Advance the model by `dt` seconds under `input`.
    pub fn step(&mut self, input: PointerInput, dt: f64) {
        let dt = dt.clamp(0.0, 0.1);
        if input.engaged {
            self.zoom(input, dt);
        }
        self.climb_out_if_outside();
        self.maintain_tree();
        self.update_commit(dt, input.engaged);
        self.advance_frame_if_deep();
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
        let s = e.sym as usize;
        let (c_lo, c_hi) = (e.cuts[s], e.cuts[s + 1]);
        let cw = c_hi - c_lo;
        let n_sym = self.alphabet.len();
        let mut dst: Vec<Node> = Vec::with_capacity(self.nodes.len() + n_sym + 1);
        dst.push(Node {
            sym: 0,
            lo: 0.0,
            hi: 1.0,
            depth: 0,
            parent: ROOT,
            children: Vec::new(),
            expanded: true,
            alive: true,
        });
        let mut kids = Vec::with_capacity(n_sym);
        for i in 0..n_sym {
            if i == s {
                let idx = copy_subtree(&self.nodes, ROOT, &mut dst, ROOT, 1, c_lo, cw);
                dst[idx].sym = e.sym;
                dst[idx].lo = c_lo;
                dst[idx].hi = c_hi;
                kids.push(idx);
            } else {
                let idx = dst.len();
                dst.push(Node {
                    sym: i as u8,
                    lo: e.cuts[i],
                    hi: e.cuts[i + 1],
                    depth: 1,
                    parent: ROOT,
                    children: Vec::new(),
                    expanded: false,
                    alive: true,
                });
                kids.push(idx);
            }
        }
        dst[ROOT].children = kids;
        self.nodes = dst;
        self.view_lo = c_lo + self.view_lo * cw;
        self.view_hi = c_lo + self.view_hi * cw;
    }

    /// Deep-zoom rebase (forward): once the view is tiny relative to the
    /// frame, descend the root into committed children. Pure re-anchoring —
    /// the ledger records each level's exact partition for the way back.
    fn advance_frame_if_deep(&mut self) {
        while self.view_hi - self.view_lo < self.params.rebase_view_w && self.tail_len() > 0 {
            if !self.advance_frame_once() {
                return;
            }
        }
    }

    fn advance_frame_once(&mut self) -> bool {
        if !self.nodes[ROOT].expanded {
            return false;
        }
        let sym = self.committed[self.ledger.len()];
        let child = self.nodes[ROOT].children[sym as usize];
        let (c_lo, c_hi) = (self.nodes[child].lo, self.nodes[child].hi);
        // Only advance into a child that holds the view (mid-retraction the
        // committed child may not; climbing right back out would be churn).
        if !(c_lo <= self.view_lo && self.view_hi <= c_hi) {
            return false;
        }
        let mut cuts: Vec<f64> = self.nodes[ROOT]
            .children
            .iter()
            .map(|&i| self.nodes[i].lo)
            .collect();
        cuts.push(1.0);
        self.ledger.push(LedgerEntry { cuts: cuts.into(), sym });
        let scale = 1.0 / (c_hi - c_lo);
        let offset = -c_lo * scale;
        let mut dst = Vec::with_capacity(self.nodes.len());
        copy_subtree(&self.nodes, child, &mut dst, ROOT, 0, offset, scale);
        dst[ROOT].lo = 0.0;
        dst[ROOT].hi = 1.0;
        self.nodes = dst;
        self.view_lo = offset + self.view_lo * scale;
        self.view_hi = offset + self.view_hi * scale;
        true
    }

    /// Lazy, screen-driven expansion and eviction (SPEC §5.1). Thresholds are
    /// pixels: screen size *is* cumulative probability × zoom.
    fn maintain_tree(&mut self) {
        if self.nodes.len() > self.params.max_nodes {
            self.compact();
        }
        let w = self.view_hi - self.view_lo;
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
                && depth < self.params.max_depth
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
        let (lo, hi, depth) = {
            let n = &self.nodes[i];
            (n.lo, n.hi, n.depth)
        };
        let ctx = self.context_of(i);
        let dist = self.lm.predict(&ctx);
        debug_assert_eq!(dist.len(), self.alphabet.len());
        let w = hi - lo;
        let mut cursor = lo;
        let mut children = Vec::with_capacity(dist.len());
        for (sym, &p) in dist.iter().enumerate() {
            let c_lo = cursor;
            // Last child absorbs rounding so children exactly partition [lo,hi).
            let c_hi = if sym + 1 == dist.len() { hi } else { cursor + p * w };
            cursor = c_hi;
            children.push(self.nodes.len());
            self.nodes.push(Node {
                sym: sym as u8,
                lo: c_lo,
                hi: c_hi,
                depth: depth + 1,
                parent: i,
                children: Vec::new(),
                expanded: false,
                alive: true,
            });
        }
        let n = &mut self.nodes[i];
        n.children = children;
        n.expanded = true;
    }

    fn evict_children(&mut self, i: usize) {
        let mut stack = std::mem::take(&mut self.nodes[i].children);
        self.nodes[i].expanded = false;
        while let Some(c) = stack.pop() {
            self.nodes[c].alive = false;
            stack.extend(std::mem::take(&mut self.nodes[c].children));
        }
    }

    /// Frame-absorbed text plus path symbols from frame root to node `i`.
    fn context_of(&self, i: usize) -> Vec<u8> {
        let mut path = Vec::new();
        let mut cur = i;
        while cur != ROOT {
            path.push(self.nodes[cur].sym);
            cur = self.nodes[cur].parent;
        }
        let mut ctx = self.committed[..self.ledger.len()].to_vec();
        ctx.extend(path.iter().rev());
        ctx
    }

    /// Chain of nodes (root-child first) that fully contain the view.
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
    /// will be written if the user keeps steering here.
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

    /// Uncommitted symbols under the crosshair, for the ribbon preview.
    /// Empty while the crosshair path disagrees with the committed tail
    /// (i.e. mid-retraction).
    pub fn pending_syms(&self) -> Vec<u8> {
        let tail = &self.committed[self.ledger.len()..];
        let path = self.center_path();
        if path.len() < tail.len() {
            return Vec::new();
        }
        let syms: Vec<u8> = path.iter().map(|&i| self.nodes[i].sym).collect();
        if !syms.starts_with(tail) {
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
        let eps = RETRACT_EPS * (self.view_hi - self.view_lo);
        let mut cur = ROOT;
        for &s in &self.committed[self.ledger.len()..] {
            if !self.nodes[cur].expanded {
                return false;
            }
            cur = self.nodes[cur].children[s as usize];
        }
        let n = &self.nodes[cur];
        n.lo - eps <= self.view_lo && self.view_hi <= n.hi + eps
    }

    /// Debounced text watermark, both directions (SPEC §5.2/§8.3): a node
    /// must contain the view for the debounce period to commit; the view must
    /// stay outside the deepest committed node for the same period to retract
    /// one symbol. Geometry never changes here — only the buffer and the LM.
    fn update_commit(&mut self, dt: f64, engaged: bool) {
        let chain = self.contained_path();
        let tail = self.tail_len();
        let on_track = chain.len() >= tail
            && chain[..tail]
                .iter()
                .map(|&i| self.nodes[i].sym)
                .eq(self.committed[self.ledger.len()..].iter().copied());
        if on_track && chain.len() > tail {
            self.outside_timer = 0.0;
            let cand = self.nodes[chain[tail]].sym;
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
            for &id in &chain[tail..] {
                let sym = self.nodes[id].sym;
                self.lm.learn(&self.committed, sym);
                self.committed.push(sym);
            }
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
        }
    }

    /// Interval, in current-frame coordinates, of a continuation `next`
    /// beyond the committed text. Follows actual tree geometry where nodes
    /// exist (the LM may have learned since a node was laid out) and extends
    /// with fresh LM predictions below the tree. Used by the oracle sim.
    pub fn path_interval(&mut self, next: &[u8]) -> (f64, f64) {
        let mut syms: Vec<u8> = self.committed[self.ledger.len()..].to_vec();
        syms.extend_from_slice(next);
        let mut ctx: Vec<u8> = self.committed[..self.ledger.len()].to_vec();
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        let mut node = Some(ROOT);
        for &s in &syms {
            match node {
                Some(i) if self.nodes[i].expanded => {
                    let c = self.nodes[i].children[s as usize];
                    lo = self.nodes[c].lo;
                    hi = self.nodes[c].hi;
                    node = Some(c);
                }
                _ => {
                    node = None;
                    let dist = self.lm.predict(&ctx);
                    let w = hi - lo;
                    let before: f64 = dist[..s as usize].iter().sum();
                    let n_lo = lo + w * before;
                    hi = n_lo + w * dist[s as usize];
                    lo = n_lo;
                }
            }
            ctx.push(s);
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
            .map(|(i, n)| BoxView {
                sym: n.sym,
                depth: frame_depth + n.depth,
                y0: (n.lo - self.view_lo) / w,
                y1: (n.hi - self.view_lo) / w,
                on_path: path.contains(&i),
            })
            .collect();
        out.sort_by(|a, b| a.depth.cmp(&b.depth).then(a.y0.total_cmp(&b.y0)));
        out
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
                PointerInput { dx: -1.0, y: 0.5, engaged: true }
            } else {
                let rest = &target[committed.len()..];
                let depth = rest.len().min(4);
                let (lo, hi) = e.path_interval(&rest[..depth]);
                let (v_lo, v_hi) = e.view();
                let y = ((lo + hi) / 2.0 - v_lo) / (v_hi - v_lo);
                if (0.0..=1.0).contains(&y) {
                    PointerInput { dx: 1.0, y, engaged: true }
                } else {
                    PointerInput { dx: -0.3, y: y.clamp(0.0, 1.0), engaged: true }
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
        e.step(PointerInput { dx: 0.0, y: 0.5, engaged: false }, 1.0 / 60.0);
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
            e.step(PointerInput { dx: -1.0, y: 0.5, engaged: true }, 1.0 / 60.0);
            if e.committed_syms().is_empty() {
                break;
            }
        }
        assert!(e.committed_syms().is_empty(), "retraction should undo commit");
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
                e.step(PointerInput { dx: 0.9, y, engaged: true }, 1.0 / 60.0);
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
            e.step(PointerInput { dx: -1.0, y: 0.1, engaged: true }, 1.0 / 60.0);
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
            e.step(PointerInput { dx: 0.8, y: 0.5, engaged: true }, 1.0 / 60.0);
        }
        let (lo0, hi0) = e.view();
        let (c0, w0) = ((lo0 + hi0) / 2.0, hi0 - lo0);
        for _ in 0..60 {
            e.step(PointerInput { dx: 0.0, y: 0.1, engaged: true }, 1.0 / 60.0);
        }
        let (lo1, hi1) = e.view();
        let c1 = (lo1 + hi1) / 2.0;
        assert!(c1 < c0 - 0.1 * w0, "pointing up must scroll up: {c0} -> {c1}");
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
        e.step(PointerInput { dx: 0.0, y: 0.5, engaged: false }, 1.0 / 60.0);
        let cuts0: Vec<f64> = e.nodes[ROOT].children.iter().map(|&c| e.nodes[c].lo).collect();

        let target = [1u8, 0, 2]; // "bac"
        assert!(steer_to(&mut e, &target, 3000), "should write the target");
        assert!(!e.ledger.is_empty(), "low rebase threshold must advance the frame");

        // Zoom all the way back out; text retracts and frames climb.
        for _ in 0..3000 {
            e.step(PointerInput { dx: -1.0, y: 0.5, engaged: true }, 1.0 / 60.0);
            if e.committed_syms().is_empty() && e.ledger.is_empty() {
                break;
            }
        }
        assert!(e.committed_syms().is_empty(), "full retraction");
        assert!(e.ledger.is_empty(), "frame back at document root");
        let cuts1: Vec<f64> = e.nodes[ROOT].children.iter().map(|&c| e.nodes[c].lo).collect();
        assert_eq!(cuts0, cuts1, "climb-out must restore geometry exactly");
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
            e.step(PointerInput { dx: 1.0, y, engaged: true }, 1.0 / 60.0);
            if e.committed_syms().is_empty() && e.pending_syms().first() == Some(&0) {
                saw_preview = true;
            }
            if !e.committed_syms().is_empty() {
                break;
            }
        }
        assert!(saw_preview, "pending path must show 'a' before it commits");
    }
}

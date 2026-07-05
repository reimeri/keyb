//! Interval tree + steering dynamics + commit/retract (SPEC §5).
//!
//! Text space is the unit interval [0,1) of the *current frame*. The frame is
//! rebased on every commit so f64 never sees deep-zoom precision loss.

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
    /// Expand a node once its projected height exceeds this (px).
    pub expand_px: f64,
    /// Evict children once a node's projected height falls below this (px).
    pub evict_px: f64,
    /// A node must fully contain the view this long before committing (s).
    pub commit_debounce: f64,
    /// Expansion depth limit below the current frame root.
    pub max_depth: usize,
    /// Arena size that triggers a rebuild (garbage collection).
    pub max_nodes: usize,
}

impl Default for EngineParams {
    fn default() -> Self {
        EngineParams {
            max_bits_per_sec: 4.5,
            expand_px: 40.0,
            evict_px: 2.0,
            commit_debounce: 0.15,
            max_depth: 12,
            max_nodes: 4000,
        }
    }
}

struct Node {
    sym: u8,
    lo: f64,
    hi: f64,
    depth: usize,
    parent: usize,
    children: Vec<usize>,
    expanded: bool,
    alive: bool,
}

/// One committed symbol: its interval within its parent's frame, so backward
/// zoom can restore the parent frame exactly (SPEC §8.3 retraction ledger,
/// internal-buffer version for M1).
#[derive(Clone, Copy, Debug)]
struct LedgerEntry {
    lo: f64,
    hi: f64,
}

/// A renderable box: vertical extent as fractions of the canvas (may exceed
/// [0,1]; the renderer clips).
#[derive(Clone, Copy, Debug)]
pub struct BoxView {
    pub sym: u8,
    pub depth: usize,
    pub y0: f64,
    pub y1: f64,
}

pub struct Engine {
    pub alphabet: Alphabet,
    lm: Box<dyn LanguageModel>,
    pub params: EngineParams,
    nodes: Vec<Node>,
    view_lo: f64,
    view_hi: f64,
    committed: Vec<u8>,
    ledger: Vec<LedgerEntry>,
    commit_candidate: Option<u8>, // symbol of first contained child
    commit_timer: f64,
    outside_timer: f64,
    viewport_px: f64,
    /// Current speed cap in bits/s; UI may adjust, sim sets directly.
    pub speed_bits: f64,
}

const ROOT: usize = 0;
/// View may exceed the frame by this fraction of its width before a
/// retraction fires (overshoot hysteresis).
const RETRACT_EPS: f64 = 0.02;
const MIN_VIEW_W: f64 = 1e-9;
const MAX_VIEW_W: f64 = 4.0;

impl Engine {
    pub fn new(alphabet: Alphabet, lm: Box<dyn LanguageModel>, params: EngineParams) -> Self {
        assert_eq!(lm.alphabet_len(), alphabet.len());
        let mut e = Engine {
            alphabet,
            lm,
            params,
            nodes: Vec::new(),
            view_lo: 0.0,
            view_hi: 1.0,
            committed: Vec::new(),
            ledger: Vec::new(),
            commit_candidate: None,
            commit_timer: 0.0,
            outside_timer: 0.0,
            viewport_px: 800.0,
            speed_bits: params.max_bits_per_sec,
        };
        e.rebuild_root();
        e
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

    /// Advance the model by `dt` seconds under `input`.
    pub fn step(&mut self, input: PointerInput, dt: f64) {
        let dt = dt.clamp(0.0, 0.1);
        if input.engaged {
            self.zoom(input, dt);
        }
        self.retract_if_outside(dt);
        self.maintain_tree();
        self.check_commit(dt, input.engaged);
    }

    fn zoom(&mut self, input: PointerInput, dt: f64) {
        let w = self.view_hi - self.view_lo;
        let dx = input.dx.clamp(-1.0, 1.0);
        let y = input.y.clamp(-0.5, 1.5);
        let new_w = (w * (-self.speed_bits * dx * dt).exp2()).clamp(MIN_VIEW_W, MAX_VIEW_W);
        // The point under the pointer stays at the same view fraction.
        let p = self.view_lo + y * w;
        self.view_lo = p - y * new_w;
        self.view_hi = self.view_lo + new_w;
    }

    /// Leaving the current frame means un-committing (SPEC §5.2: backward
    /// zoom is the only undo). Retraction is debounced like commit — the view
    /// must stay outside the frame for the debounce period before one level
    /// pops; otherwise transient overshoot (and delayed/noisy steering)
    /// causes a commit/retract limit cycle. With an empty ledger the view is
    /// clamped to the document root instead.
    fn retract_if_outside(&mut self, dt: f64) {
        let w = self.view_hi - self.view_lo;
        let eps = RETRACT_EPS * w;
        let outside = self.view_lo < -eps || self.view_hi > 1.0 + eps;
        if !outside {
            self.outside_timer = 0.0;
            return;
        }
        if self.ledger.is_empty() {
            let mut w = (self.view_hi - self.view_lo).min(1.0);
            w = w.max(MIN_VIEW_W);
            let mut lo = self.view_lo.max(0.0);
            if lo + w > 1.0 {
                lo = 1.0 - w;
            }
            self.view_lo = lo;
            self.view_hi = lo + w;
            self.outside_timer = 0.0;
            return;
        }
        self.outside_timer += dt;
        if self.outside_timer < self.params.commit_debounce {
            return;
        }
        self.outside_timer = 0.0;
        // Pop exactly one level; mapping into the parent frame shrinks the
        // view by the entry's width, so multi-level exits retract one
        // debounced level at a time.
        let entry = self.ledger.pop().expect("checked non-empty");
        self.committed.pop();
        let pw = entry.hi - entry.lo;
        self.view_lo = entry.lo + self.view_lo * pw;
        self.view_hi = entry.lo + self.view_hi * pw;
        self.rebuild_root();
    }

    fn rebuild_root(&mut self) {
        self.nodes.clear();
        self.nodes.push(Node {
            sym: 0,
            lo: 0.0,
            hi: 1.0,
            depth: 0,
            parent: ROOT,
            children: Vec::new(),
            expanded: false,
            alive: true,
        });
        self.commit_candidate = None;
        self.commit_timer = 0.0;
    }

    /// Lazy, screen-driven expansion and eviction (SPEC §5.1). Thresholds are
    /// pixels: screen size *is* cumulative probability × zoom.
    fn maintain_tree(&mut self) {
        if self.nodes.len() > self.params.max_nodes {
            self.rebuild_root();
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

    /// Committed text plus path symbols from frame root to node `i`.
    fn context_of(&self, i: usize) -> Vec<u8> {
        let mut path = Vec::new();
        let mut cur = i;
        while cur != ROOT {
            path.push(self.nodes[cur].sym);
            cur = self.nodes[cur].parent;
        }
        let mut ctx = self.committed.clone();
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

    /// Commit once a node has contained the view for the debounce period
    /// (SPEC §8.3). The whole contained chain commits at once — deeper
    /// containment implies the first link was contained throughout.
    fn check_commit(&mut self, dt: f64, engaged: bool) {
        let path = self.contained_path();
        let first_sym = path.first().map(|&i| self.nodes[i].sym);
        if first_sym.is_none() || first_sym != self.commit_candidate {
            self.commit_candidate = first_sym;
            self.commit_timer = 0.0;
            return;
        }
        if engaged {
            self.commit_timer += dt;
        }
        if self.commit_timer < self.params.commit_debounce {
            return;
        }
        // Commit the chain top-down, recording each interval relative to its
        // parent frame for exact retraction later.
        let (mut p_lo, mut p_hi) = (0.0f64, 1.0f64);
        for &id in &path {
            let (n_lo, n_hi, sym) = {
                let n = &self.nodes[id];
                (n.lo, n.hi, n.sym)
            };
            let pw = p_hi - p_lo;
            self.ledger.push(LedgerEntry {
                lo: (n_lo - p_lo) / pw,
                hi: (n_hi - p_lo) / pw,
            });
            self.lm.learn(&self.committed, sym);
            self.committed.push(sym);
            p_lo = n_lo;
            p_hi = n_hi;
        }
        // Rebase: the deepest committed node becomes the new [0,1) frame.
        let fw = p_hi - p_lo;
        self.view_lo = (self.view_lo - p_lo) / fw;
        self.view_hi = (self.view_hi - p_lo) / fw;
        self.rebuild_root();
    }

    /// Interval of a hypothetical continuation below the current frame root,
    /// computed straight from the LM (used by the oracle sim and, later, by
    /// tests). Matches `expand`'s partitioning exactly.
    pub fn path_interval(&mut self, next: &[u8]) -> (f64, f64) {
        let mut ctx = self.committed.clone();
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for &s in next {
            let dist = self.lm.predict(&ctx);
            let w = hi - lo;
            let before: f64 = dist[..s as usize].iter().sum();
            let n_lo = lo + w * before;
            let n_hi = n_lo + w * dist[s as usize];
            lo = n_lo;
            hi = n_hi;
            ctx.push(s);
        }
        (lo, hi)
    }

    /// Boxes to draw, parents before children.
    pub fn visible(&self) -> Vec<BoxView> {
        let w = self.view_hi - self.view_lo;
        let mut out: Vec<BoxView> = self
            .nodes
            .iter()
            .skip(1)
            .filter(|n| n.alive && n.hi > self.view_lo && n.lo < self.view_hi)
            .map(|n| BoxView {
                sym: n.sym,
                depth: n.depth,
                y0: (n.lo - self.view_lo) / w,
                y1: (n.hi - self.view_lo) / w,
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
        // Steer into the middle of the first child ('a' under uniform).
        let (lo, hi) = e.path_interval(&[0]);
        let target = (lo + hi) / 2.0;
        for _ in 0..600 {
            let (vlo, vhi) = e.view();
            let y = (target - vlo) / (vhi - vlo);
            e.step(PointerInput { dx: 1.0, y, engaged: true }, 1.0 / 60.0);
            if !e.committed_syms().is_empty() {
                break;
            }
        }
        assert_eq!(e.committed_syms(), &[0], "should have committed 'a'");
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
}

//! The alphabet is two layers (SPEC §5.4):
//!
//! - a flat, fixed-order list of **leaf symbols** — the writable characters.
//!   This is the *only* thing the language model sees: it predicts a
//!   distribution over leaves, and committed text is a sequence of leaves.
//! - a **layout** grouping those leaves for display: frequent letters sit at
//!   the top level, while rarer runs (capitals, digits, symbols) live inside
//!   expandable group boxes, exactly as in classic Dasher.
//!
//! Grouping is a pure structuring/rendering concern: a leaf's cumulative
//! interval in `[0,1)` is `P(leaf)` whether or not it sits inside a group, so
//! arithmetic-coding geometry — and therefore steering feel and the oracle's
//! WPM on ungrouped lowercase text — is unchanged. The layout only inserts
//! intermediate container boxes on the path to a grouped leaf.
//!
//! The leaf ordering is fixed forever (SPEC §5.4); only sizes change.

use std::collections::HashMap;

/// Distinct, high-contrast color families for the special group boxes so they
/// pop out from the per-letter palette (SPEC §5.4). The renderer maps these to
/// concrete hues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupColor {
    Capitals,
    Digits,
    Symbols,
}

/// One entry in a layout node's ordered child list.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Item {
    /// A writable leaf symbol (index into `Alphabet::leaves`).
    Leaf(u8),
    /// An expandable group box (index into `Layout::nodes`).
    Group(u32),
}

/// A node of the layout tree: an ordered list of items. Node 0 is the root
/// (the choices at every fresh character position). Group nodes carry a
/// display label and color.
struct LNode {
    items: Vec<Item>,
    label: Option<char>,
    color: Option<GroupColor>,
    /// Every leaf reachable in this node's subtree (for aggregate group mass).
    leaves: Vec<u8>,
    /// leaf symbol -> (item index within `items`, `true` if that item *is* the
    /// leaf, `false` if it is the group containing the leaf). O(1) descent.
    leaf_pos: HashMap<u8, (usize, bool)>,
}

struct Layout {
    nodes: Vec<LNode>,
}

/// Fixed, ordered set of writable symbols plus their display grouping.
///
/// The ordering is part of the user's spatial memory and must never change at
/// runtime (SPEC §5.4). Sizes change with probability; positions in the column
/// order do not.
pub struct Alphabet {
    leaves: Vec<char>,
    layout: Layout,
    /// Minimum share of a node's interval each group box is guaranteed,
    /// regardless of how little probability its members carry (SPEC §5.4's
    /// "configured prior"). Without this the special groups collapse to the
    /// smoothing floor (~3 px) under a lowercase-biased model and can't be
    /// aimed at; this keeps them steerable.
    group_prior: f64,
}

impl Clone for Alphabet {
    fn clone(&self) -> Self {
        // Rebuild from the leaf list + a structural description so the derived
        // caches (leaf_pos, leaves) never drift. Cheap and only done on the few
        // clones the LM crates make at startup.
        Alphabet::from_spec(self.leaves.clone(), &self.spec()).with_group_prior(self.group_prior)
    }
}

impl std::fmt::Debug for Alphabet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Alphabet")
            .field("leaves", &self.leaves.len())
            .field("groups", &(self.layout.nodes.len() - 1))
            .finish()
    }
}

/// A structural description of the layout, independent of the built caches —
/// the single source both constructors and `Clone` build from.
struct GroupSpec {
    label: char,
    color: GroupColor,
    leaves: std::ops::Range<u8>,
}
struct LayoutSpec {
    /// Leaf symbols that stay at the top level, in order.
    top_leaves: std::ops::Range<u8>,
    /// Group boxes appended after the top-level leaves.
    groups: Vec<GroupSpec>,
}

impl Alphabet {
    /// v1 English alphabet: a–z, space, then a small punctuation tail. Flat
    /// (no groups) — the shape the engine and LM tests are written against.
    pub fn english_v1() -> Self {
        let mut leaves: Vec<char> = ('a'..='z').collect();
        leaves.push(' ');
        leaves.extend(['.', ',', '\'']);
        let n = leaves.len() as u8;
        Alphabet::from_spec(
            leaves,
            &LayoutSpec {
                top_leaves: 0..n,
                groups: Vec::new(),
            },
        )
    }

    /// Full English alphabet for interactive use: the v1 leaves (identical
    /// order, so v1 code paths are a strict prefix) followed by capitals,
    /// digits, and an extended symbol set — each behind its own group box.
    pub fn english_full() -> Self {
        // Top-level, in order: a–z, space, then the three most common
        // punctuation marks. Indices 0..30 match `english_v1` exactly.
        let mut leaves: Vec<char> = ('a'..='z').collect();
        leaves.push(' ');
        leaves.extend(['.', ',', '\'']);
        let top_end = leaves.len() as u8; // 30

        let caps_start = leaves.len() as u8;
        leaves.extend('A'..='Z');
        let caps_end = leaves.len() as u8;

        let digits_start = leaves.len() as u8;
        leaves.extend('0'..='9');
        let digits_end = leaves.len() as u8;

        let sym_start = leaves.len() as u8;
        leaves.extend([
            '!', '?', ';', ':', '"', '-', '(', ')', '/', '@', '#', '&', '*', '+', '=', '_', '%',
        ]);
        let sym_end = leaves.len() as u8;

        let spec = LayoutSpec {
            top_leaves: 0..top_end,
            groups: vec![
                GroupSpec {
                    label: 'A',
                    color: GroupColor::Capitals,
                    leaves: caps_start..caps_end,
                },
                GroupSpec {
                    label: '1',
                    color: GroupColor::Digits,
                    leaves: digits_start..digits_end,
                },
                GroupSpec {
                    label: '!',
                    color: GroupColor::Symbols,
                    leaves: sym_start..sym_end,
                },
            ],
        };
        Alphabet::from_spec(leaves, &spec)
    }

    /// Recover the structural description (for `Clone`). Assumes the layout is
    /// the one-level "top leaves then contiguous groups" shape both
    /// constructors produce, which is all this crate builds.
    fn spec(&self) -> LayoutSpec {
        let root = &self.layout.nodes[0];
        let mut top_end = 0u8;
        let mut groups = Vec::new();
        for &item in &root.items {
            match item {
                Item::Leaf(s) => top_end = s + 1,
                Item::Group(g) => {
                    let n = &self.layout.nodes[g as usize];
                    let lo = *n.leaves.iter().min().expect("group has leaves");
                    let hi = *n.leaves.iter().max().expect("group has leaves") + 1;
                    groups.push(GroupSpec {
                        label: n.label.expect("group has a label"),
                        color: n.color.expect("group has a color"),
                        leaves: lo..hi,
                    });
                }
            }
        }
        LayoutSpec {
            top_leaves: 0..top_end,
            groups,
        }
    }

    fn from_spec(leaves: Vec<char>, spec: &LayoutSpec) -> Self {
        // Build group nodes first (ids 1..), then the root (id 0) referring to
        // them. We assemble into a temporary vec and rotate the root to front.
        let mut group_nodes: Vec<LNode> = Vec::new();
        let mut root_items: Vec<Item> = spec.top_leaves.clone().map(Item::Leaf).collect();
        for g in &spec.groups {
            // Group node id = its position among groups, +1 for the root slot.
            let gid = (group_nodes.len() + 1) as u32;
            let members: Vec<u8> = g.leaves.clone().collect();
            let items: Vec<Item> = members.iter().copied().map(Item::Leaf).collect();
            group_nodes.push(LNode {
                items,
                label: Some(g.label),
                color: Some(g.color),
                leaves: members,
                leaf_pos: HashMap::new(),
            });
            root_items.push(Item::Group(gid));
        }
        let root_leaves: Vec<u8> = (0..leaves.len() as u8).collect();
        let root = LNode {
            items: root_items,
            label: None,
            color: None,
            leaves: root_leaves,
            leaf_pos: HashMap::new(),
        };
        let mut nodes = Vec::with_capacity(group_nodes.len() + 1);
        nodes.push(root);
        nodes.append(&mut group_nodes);
        // Fill the O(1) descent maps now that every node exists.
        for nid in 0..nodes.len() {
            let mut map = HashMap::new();
            for (k, item) in nodes[nid].items.iter().enumerate() {
                match *item {
                    Item::Leaf(s) => {
                        map.insert(s, (k, true));
                    }
                    Item::Group(g) => {
                        for &leaf in &nodes[g as usize].leaves {
                            map.insert(leaf, (k, false));
                        }
                    }
                }
            }
            nodes[nid].leaf_pos = map;
        }
        // ~2% of the parent interval per group — a ~15 px box on a typical
        // column that grows as you steer toward it, rather than a 3 px sliver.
        Alphabet {
            leaves,
            layout: Layout { nodes },
            group_prior: 0.02,
        }
    }

    /// Override the per-group minimum size (SPEC §5.4). 0 disables it.
    pub fn with_group_prior(mut self, prior: f64) -> Self {
        self.group_prior = prior.clamp(0.0, 0.2);
        self
    }

    /// Number of leaf symbols — the language model's alphabet size.
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    pub fn char_of(&self, sym: u8) -> char {
        self.leaves[sym as usize]
    }

    pub fn index_of(&self, c: char) -> Option<u8> {
        self.leaves.iter().position(|&x| x == c).map(|i| i as u8)
    }

    /// Display label for a leaf symbol (space gets a visible mark).
    pub fn label(&self, sym: u8) -> char {
        match self.char_of(sym) {
            ' ' => '␣',
            c => c,
        }
    }

    /// Map a string to leaf symbols. A char maps to its own leaf if present,
    /// else to its lowercase form's leaf (so lowercase-only alphabets still
    /// absorb capitalized input), else it is skipped.
    pub fn encode(&self, text: &str) -> Vec<u8> {
        let mut out = Vec::new();
        for c in text.chars() {
            if let Some(s) = self.index_of(c) {
                out.push(s);
            } else if let Some(s) = c.to_lowercase().next().and_then(|l| self.index_of(l)) {
                out.push(s);
            }
        }
        out
    }

    pub fn decode(&self, syms: &[u8]) -> String {
        syms.iter().map(|&s| self.char_of(s)).collect()
    }

    // --- Layout access (engine-internal) ---------------------------------

    /// The root layout node id — the choices at every fresh character position.
    pub(crate) fn root_layout(&self) -> u32 {
        0
    }

    /// Items of layout node `lid`, in fixed order.
    pub(crate) fn items(&self, lid: u32) -> &[Item] {
        &self.layout.nodes[lid as usize].items
    }

    /// Layout node a given item opens into when expanded: a group opens into
    /// itself; a leaf opens into the root (the next character position).
    pub(crate) fn opens_of(&self, item: Item) -> u32 {
        match item {
            Item::Leaf(_) => self.root_layout(),
            Item::Group(g) => g,
        }
    }

    /// Per-item aggregate weight for partitioning: a leaf's own probability, a
    /// group's summed member probability (floored at `group_prior` so the
    /// special boxes stay steerable). `dist` is P(leaf) over all leaves; the
    /// caller renormalizes by the returned sum.
    pub(crate) fn item_weights(&self, lid: u32, dist: &[f64]) -> Vec<f64> {
        self.layout.nodes[lid as usize]
            .items
            .iter()
            .map(|item| match *item {
                Item::Leaf(s) => dist[s as usize],
                Item::Group(g) => {
                    let mass: f64 = self.layout.nodes[g as usize]
                        .leaves
                        .iter()
                        .map(|&l| dist[l as usize])
                        .sum();
                    mass.max(self.group_prior)
                }
            })
            .collect()
    }

    /// Within layout node `lid`, the item leading toward leaf `c`, and whether
    /// that item *is* the leaf (`true`) or the group containing it (`false`).
    pub(crate) fn item_leading_to(&self, lid: u32, c: u8) -> Option<(usize, bool)> {
        self.layout.nodes[lid as usize].leaf_pos.get(&c).copied()
    }

    pub(crate) fn group_label(&self, gid: u32) -> char {
        self.layout.nodes[gid as usize].label.unwrap_or('?')
    }

    pub(crate) fn group_color(&self, gid: u32) -> GroupColor {
        self.layout.nodes[gid as usize]
            .color
            .unwrap_or(GroupColor::Symbols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_is_flat() {
        let a = Alphabet::english_v1();
        assert_eq!(a.len(), 30);
        // Every root item is a leaf, in symbol order.
        for (k, item) in a.items(a.root_layout()).iter().enumerate() {
            match item {
                Item::Leaf(s) => assert_eq!(*s as usize, k),
                Item::Group(_) => panic!("v1 must be flat"),
            }
        }
    }

    #[test]
    fn full_prefixes_v1() {
        let v1 = Alphabet::english_v1();
        let full = Alphabet::english_full();
        for s in 0..v1.len() as u8 {
            assert_eq!(v1.char_of(s), full.char_of(s), "leaf {s} must match");
        }
    }

    #[test]
    fn full_has_three_groups_and_reaches_grouped_leaves() {
        let a = Alphabet::english_full();
        let root = a.root_layout();
        let groups = a
            .items(root)
            .iter()
            .filter(|i| matches!(i, Item::Group(_)))
            .count();
        assert_eq!(groups, 3, "capitals, digits, symbols");
        // 'A' is inside a group, not at top level.
        let a_sym = a.index_of('A').unwrap();
        let (_, is_leaf_at_root) = a.item_leading_to(root, a_sym).unwrap();
        assert!(!is_leaf_at_root, "'A' must sit inside a group");
        // 'a' is a top-level leaf.
        let low = a.index_of('a').unwrap();
        let (_, is_leaf) = a.item_leading_to(root, low).unwrap();
        assert!(is_leaf, "'a' must be a top-level leaf");
    }

    #[test]
    fn item_weights_aggregate_groups() {
        let a = Alphabet::english_full();
        let dist = vec![1.0 / a.len() as f64; a.len()];
        let w = a.item_weights(a.root_layout(), &dist);
        // Total weight is conserved.
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "weights partition unity: {sum}");
        // The capitals group weight equals 26 leaves' worth.
        let cap_item = a
            .items(a.root_layout())
            .iter()
            .position(|i| matches!(i, Item::Group(g) if a.group_color(*g) == GroupColor::Capitals))
            .unwrap();
        assert!((w[cap_item] - 26.0 / a.len() as f64).abs() < 1e-12);
    }

    #[test]
    fn encode_preserves_case_when_available() {
        let full = Alphabet::english_full();
        assert_eq!(
            full.encode("Ab"),
            vec![full.index_of('A').unwrap(), full.index_of('b').unwrap()]
        );
        // Lowercase-only alphabet folds capitals down.
        let v1 = Alphabet::english_v1();
        assert_eq!(
            v1.encode("Ab"),
            vec![v1.index_of('a').unwrap(), v1.index_of('b').unwrap()]
        );
    }

    #[test]
    fn clone_rebuilds_equivalent_layout() {
        let a = Alphabet::english_full();
        let b = a.clone();
        assert_eq!(a.len(), b.len());
        let da = vec![1.0 / a.len() as f64; a.len()];
        assert_eq!(
            a.item_weights(a.root_layout(), &da),
            b.item_weights(b.root_layout(), &da)
        );
        let s = a.index_of('7').unwrap();
        assert_eq!(a.item_leading_to(0, s), b.item_leading_to(0, s));
    }
}

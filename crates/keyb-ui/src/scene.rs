//! Scene construction: nested boxes, labels with LOD, ribbon, origin line
//! (SPEC §7.1). Rebuilt every frame; visible node count is bounded by the
//! engine's expand/evict thresholds.

use std::collections::HashMap;

use keyb_core::{Engine, PointerInput};
use vello::kurbo::{Affine, Line, Rect, RoundedRect, Stroke};
use vello::peniko::{Color, Fill};
use vello::{Glyph, Scene};

use crate::font::FontAssets;
use crate::{ORIGIN_X_FRAC, RIBBON_H};

/// Labels appear once a box is at least this tall (px).
const LABEL_MIN_PX: f64 = 18.0;
/// Extra gap (px) reserved between a box's label glyph and its nearest child's
/// left edge, so the glyph is never touched by the child's fill.
const LABEL_PAD: f64 = 4.0;

/// A box ready to draw: geometry, and its label (glyphs + size + baseline) if
/// tall enough. `x0` is filled in by the width recurrence.
struct BoxDraw {
    sym: u8,
    depth: usize,
    on_path: bool,
    h: f64,
    cy0: f64,
    cy1: f64,
    /// Glyph advance of the label (0 when unlabeled) — the width to protect.
    adv: f64,
    label: Option<(Vec<Glyph>, f32, f64)>,
    x0: f64,
}

/// Left edge (`x0`, logical px) for each box, widened bottom-up so a box's
/// label strip is never covered by its largest child. `boxes` must be
/// depth-sorted ascending (as `Engine::visible` returns); each tuple is
/// `(id, parent, h, adv, has_label)` with `h` the box height and `adv` its
/// label advance in px. Output is aligned with the input order.
///
/// A box's fill spans `[x0, lw]`, and a descendant's fill x-range is always a
/// subset of its parent's, so the only box that can reach into a parent's
/// label strip is its nearest (smallest-`x0`) immediate child. We therefore
/// reserve `margin` px to the left of that child's *actual* (already-widened)
/// edge — hence the deepest-first pass.
fn box_left_edges(boxes: &[(usize, usize, f64, f64, bool)], lw: f64) -> Vec<f64> {
    let mut x0s = vec![0.0f64; boxes.len()];
    // Left edge of the nearest child seen so far, keyed by parent id.
    let mut child_min: HashMap<usize, f64> = HashMap::new();
    for (idx, &(id, parent, h, adv, has_label)) in boxes.iter().enumerate().rev() {
        // Nominal width tracks height (∝ probability), capped at the canvas.
        let nominal_x0 = lw - (h * 1.1).min(lw * 0.98);
        let margin = if has_label {
            h.min(24.0) * 0.4 + 6.0 + adv + LABEL_PAD
        } else {
            0.0
        };
        let mut x0 = nominal_x0;
        if let Some(&cmin) = child_min.get(&id) {
            x0 = x0.min(cmin - margin);
        }
        x0 = x0.max(lw * 0.02);
        x0s[idx] = x0;
        let e = child_min.entry(parent).or_insert(f64::INFINITY);
        *e = e.min(x0);
    }
    x0s
}

pub fn build(
    scene: &mut Scene,
    engine: &Engine,
    fonts: &FontAssets,
    lw: f64,
    lh: f64,
    scale: f64,
    input: PointerInput,
) {
    // Everything is laid out in logical px, then scaled once to physical.
    let t = Affine::scale(scale);
    let canvas_h = lh - RIBBON_H;

    // Canvas background.
    scene.fill(
        Fill::NonZero,
        t,
        Color::from_rgb8(0x14, 0x15, 0x20),
        None,
        &Rect::new(0.0, RIBBON_H, lw, lh),
    );

    // Pass 1: collect drawable boxes and lay out labels (engine::visible is
    // depth-sorted ascending, so this preserves parents-before-children order).
    let mut draws: Vec<(usize, usize, BoxDraw)> = Vec::new();
    for b in engine.visible() {
        let y0 = RIBBON_H + b.y0 * canvas_h;
        let y1 = RIBBON_H + b.y1 * canvas_h;
        let h = y1 - y0;
        if h < 0.5 {
            continue;
        }
        let (cy0, cy1) = (y0.max(RIBBON_H - 2.0), y1.min(lh + 2.0));
        if cy1 <= cy0 {
            continue;
        }
        let (adv, label) = if h >= LABEL_MIN_PX {
            let text = engine.alphabet.label(b.sym).to_string();
            let font_px = (h * 0.55).clamp(13.0, 64.0) as f32;
            let (glyphs, adv) = fonts.layout(&text, font_px);
            let baseline = (y0 + y1) / 2.0 + font_px as f64 * 0.35;
            // Keep the label readable while the box edge is off-screen.
            let baseline = baseline.clamp(RIBBON_H + font_px as f64, lh - 6.0);
            (adv as f64, Some((glyphs, font_px, baseline)))
        } else {
            (0.0, None)
        };
        draws.push((
            b.id,
            b.parent,
            BoxDraw {
                sym: b.sym,
                depth: b.depth,
                on_path: b.on_path,
                h,
                cy0,
                cy1,
                adv,
                label,
                x0: 0.0,
            },
        ));
    }

    // Widen boxes leftward so each label strip stays clear of its children.
    let keys: Vec<(usize, usize, f64, f64, bool)> = draws
        .iter()
        .map(|(id, parent, d)| (*id, *parent, d.h, d.adv, d.label.is_some()))
        .collect();
    let x0s = box_left_edges(&keys, lw);
    for ((_, _, d), x0) in draws.iter_mut().zip(&x0s) {
        d.x0 = *x0;
    }

    // Pass 2: draw fills, strokes and labels, parents first.
    for (_, _, d) in draws {
        let radius = (d.h * 0.15).min(10.0);
        let rect = RoundedRect::new(d.x0, d.cy0, lw, d.cy1, radius);
        scene.fill(Fill::NonZero, t, box_color(d.sym, d.depth, d.on_path), None, &rect);
        // Crosshair-path boxes (what will be written) get a bright edge.
        let (stroke_c, stroke_w) = if d.on_path {
            (Color::from_rgba8(0xFF, 0xFF, 0xFF, 150), 1.6)
        } else {
            (Color::from_rgba8(0, 0, 0, 60), 1.0)
        };
        scene.stroke(&Stroke::new(stroke_w), t, stroke_c, None, &rect);

        if let Some((glyphs, font_px, baseline)) = d.label {
            scene
                .draw_glyphs(&fonts.font)
                .font_size(font_px)
                .transform(t * Affine::translate((d.x0 + d.h.min(24.0) * 0.4 + 6.0, baseline)))
                .brush(Color::from_rgb8(0xE8, 0xE8, 0xF0))
                .draw(Fill::NonZero, glyphs.into_iter());
        }
    }

    // Origin line + center tick (steering reference).
    let x_origin = lw * ORIGIN_X_FRAC;
    scene.stroke(
        &Stroke::new(1.0),
        t,
        Color::from_rgba8(0xFF, 0xFF, 0xFF, 70),
        None,
        &Line::new((x_origin, RIBBON_H), (x_origin, lh)),
    );
    let cy = RIBBON_H + canvas_h / 2.0;
    scene.stroke(
        &Stroke::new(2.0),
        t,
        Color::from_rgba8(0xFF, 0x60, 0x60, if input.engaged { 200 } else { 90 }),
        None,
        &Line::new((x_origin - 14.0, cy), (x_origin + 14.0, cy)),
    );

    // Ribbon: committed text tail + caret.
    scene.fill(
        Fill::NonZero,
        t,
        Color::from_rgb8(0x0C, 0x0D, 0x14),
        None,
        &Rect::new(0.0, 0.0, lw, RIBBON_H),
    );
    let font_px = 26.0f32;
    let text = engine.committed_text();
    // Pending preview: text under the crosshair that will commit if the user
    // keeps steering here — shown dim past the caret so letters appear the
    // instant they are steered over, not only after the commit debounce.
    let pending: String = engine.pending_text().chars().take(24).collect();
    let max_chars = ((lw - 60.0) / (font_px as f64 * 0.6)).max(4.0) as usize;
    let budget = max_chars.saturating_sub(pending.chars().count()).max(4);
    let tail: String = text
        .chars()
        .rev()
        .take(budget)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let shown = tail.replace(' ', "\u{2423}"); // make spaces visible in the ribbon
    let (glyphs, adv) = fonts.layout(&shown, font_px);
    let baseline = RIBBON_H * 0.68;
    scene
        .draw_glyphs(&fonts.font)
        .font_size(font_px)
        .transform(t * Affine::translate((16.0, baseline)))
        .brush(Color::from_rgb8(0xD8, 0xDC, 0xE8))
        .draw(Fill::NonZero, glyphs.into_iter());
    scene.fill(
        Fill::NonZero,
        t,
        Color::from_rgb8(0x7A, 0xA2, 0xF7),
        None,
        &Rect::new(
            16.0 + adv as f64 + 3.0,
            baseline - font_px as f64 * 0.75,
            16.0 + adv as f64 + 6.0,
            baseline + font_px as f64 * 0.15,
        ),
    );
    if !pending.is_empty() {
        let (glyphs, _) = fonts.layout(&pending.replace(' ', "\u{2423}"), font_px);
        scene
            .draw_glyphs(&fonts.font)
            .font_size(font_px)
            .transform(t * Affine::translate((16.0 + adv as f64 + 12.0, baseline)))
            .brush(Color::from_rgb8(0x6E, 0x76, 0x92))
            .draw(Fill::NonZero, glyphs.into_iter());
    }

    // Speed readout, bottom-left.
    let (glyphs, _) = fonts.layout(
        &format!("speed {:.1} bits per s   plus and minus adjust", engine.speed_bits),
        13.0,
    );
    scene
        .draw_glyphs(&fonts.font)
        .font_size(13.0)
        .transform(t * Affine::translate((12.0, lh - 10.0)))
        .brush(Color::from_rgba8(0xFF, 0xFF, 0xFF, 110))
        .draw(Fill::NonZero, glyphs.into_iter());
}

/// Golden-angle hue per symbol so neighbors in the column differ clearly;
/// depth alternates lightness (SPEC §7.1 palette placeholder). Boxes on the
/// crosshair path are lifted slightly so the pending text reads on canvas.
fn box_color(sym: u8, depth: usize, on_path: bool) -> Color {
    let hue = (sym as f64 * 137.508) % 360.0;
    let (s, mut l) = if depth % 2 == 0 { (0.42, 0.30) } else { (0.40, 0.38) };
    if on_path {
        l += 0.07;
    }
    hsl(hue, s, l)
}

fn hsl(h: f64, s: f64, l: f64) -> Color {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r, g, b) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    Color::from_rgb8(
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::{box_left_edges, LABEL_PAD};

    const LW: f64 = 620.0;

    /// Reserve reserved for a labeled box of height `h` with glyph advance `adv`.
    fn margin(h: f64, adv: f64) -> f64 {
        h.min(24.0) * 0.4 + 6.0 + adv + LABEL_PAD
    }

    /// A high-probability chain t→h→e (each nested, ~0.99 ratio) must leave
    /// every parent's label strip uncovered by its child.
    #[test]
    fn high_prob_chain_reserves_label_gutter() {
        // (id, parent, h, adv, has_label), depth-sorted ascending.
        let boxes = [
            (1usize, 0usize, 200.0, 20.0, true), // t
            (2, 1, 198.0, 20.0, true),           // h, child of t
            (3, 2, 196.0, 20.0, true),           // e, child of h
        ];
        let x0 = box_left_edges(&boxes, LW);
        // Each parent's glyph (drawn at x0 + inset) plus its advance must end
        // left of the child's left edge.
        assert!(x0[0] + margin(200.0, 20.0) <= x0[1] + 1e-6, "t under h");
        assert!(x0[1] + margin(198.0, 20.0) <= x0[2] + 1e-6, "h under e");
        // Widening only pulls boxes leftward, never past the nominal edge.
        assert!(x0[0] <= LW - 200.0 * 1.1 + 1e-6);
    }

    /// A box with only a small (low-probability) child keeps its nominal width;
    /// the child is too narrow to reach the label strip.
    #[test]
    fn small_child_leaves_parent_nominal() {
        let boxes = [
            (1usize, 0usize, 200.0, 20.0, true), // parent
            (2, 1, 30.0, 20.0, true),            // small child
        ];
        let x0 = box_left_edges(&boxes, LW);
        assert!((x0[0] - (LW - 200.0 * 1.1)).abs() < 1e-6, "parent stays nominal");
    }

    /// Left edges never cross the canvas cap even for a deep near-full chain.
    #[test]
    fn respects_left_cap() {
        let boxes = [
            (1usize, 0usize, 500.0, 20.0, true),
            (2, 1, 498.0, 20.0, true),
            (3, 2, 496.0, 20.0, true),
        ];
        let x0 = box_left_edges(&boxes, LW);
        for &e in &x0 {
            assert!(e >= LW * 0.02 - 1e-9);
        }
    }
}

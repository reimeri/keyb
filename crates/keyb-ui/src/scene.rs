//! Scene construction: nested boxes, labels with LOD, ribbon, origin line
//! (SPEC §7.1). Rebuilt every frame; visible node count is bounded by the
//! engine's expand/evict thresholds.

use keyb_core::{Engine, PointerInput};
use vello::kurbo::{Affine, Line, Rect, RoundedRect, Stroke};
use vello::peniko::{Color, Fill};
use vello::Scene;

use crate::font::FontAssets;
use crate::{ORIGIN_X_FRAC, RIBBON_H};

/// Labels appear once a box is at least this tall (px).
const LABEL_MIN_PX: f64 = 18.0;

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

    // Boxes, parents first (engine::visible is depth-sorted).
    for b in engine.visible() {
        let y0 = RIBBON_H + b.y0 * canvas_h;
        let y1 = RIBBON_H + b.y1 * canvas_h;
        let h = y1 - y0;
        if h < 0.5 {
            continue;
        }
        // Right-anchored: width tracks height (interval size), capped at the
        // canvas. A box containing the whole view fills the screen.
        let bw = (h * 1.1).min(lw * 0.98).max(6.0);
        let x0 = lw - bw;
        let (cy0, cy1) = (y0.max(RIBBON_H - 2.0), y1.min(lh + 2.0));
        if cy1 <= cy0 {
            continue;
        }
        let radius = (h * 0.15).min(10.0);
        let rect = RoundedRect::new(x0, cy0, lw, cy1, radius);
        scene.fill(Fill::NonZero, t, box_color(b.sym, b.depth), None, &rect);
        scene.stroke(
            &Stroke::new(1.0),
            t,
            Color::from_rgba8(0, 0, 0, 60),
            None,
            &rect,
        );

        if h >= LABEL_MIN_PX {
            let label = engine.alphabet.label(b.sym).to_string();
            let font_px = (h * 0.55).clamp(13.0, 64.0) as f32;
            let (glyphs, _adv) = fonts.layout(&label, font_px);
            let baseline = (y0 + y1) / 2.0 + font_px as f64 * 0.35;
            // Keep the label readable while the box edge is off-screen.
            let baseline = baseline.clamp(RIBBON_H + font_px as f64, lh - 6.0);
            scene
                .draw_glyphs(&fonts.font)
                .font_size(font_px)
                .transform(t * Affine::translate((x0 + h.min(24.0) * 0.4 + 6.0, baseline)))
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
    let max_chars = ((lw - 60.0) / (font_px as f64 * 0.6)).max(4.0) as usize;
    let tail: String = text
        .chars()
        .rev()
        .take(max_chars)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let shown = tail.replace(' ', "\u{00B7}"); // make spaces visible in the ribbon
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
/// depth alternates lightness (SPEC §7.1 palette placeholder).
fn box_color(sym: u8, depth: usize) -> Color {
    let hue = (sym as f64 * 137.508) % 360.0;
    let (s, l) = if depth % 2 == 0 { (0.42, 0.30) } else { (0.40, 0.38) };
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

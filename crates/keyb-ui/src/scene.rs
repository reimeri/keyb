//! Scene construction: nested boxes, labels with LOD, ribbon, origin line
//! (SPEC §7.1). Rebuilt every frame; visible node count is bounded by the
//! engine's expand/evict thresholds.

use std::collections::HashMap;

use keyb_core::{Engine, GroupColor, PointerInput};
use vello::kurbo::{Affine, Line, Point, Rect, RoundedRect, Stroke};
use vello::peniko::{Color, Fill, Mix};
use vello::{Glyph, Scene};

use crate::font::FontAssets;
use crate::{ORIGIN_X_FRAC, RIBBON_H};

/// Labels appear once a box is at least this tall (px).
const LABEL_MIN_PX: f64 = 18.0;
/// Left edge of ribbon text, after the reset, collapse, copy, and
/// context-capture controls.
const RIBBON_TEXT_X: f64 = 160.0;
/// Smallest canvas width at which the five ribbon controls remain separate.
pub(crate) const MIN_CANVAS_W: f64 = 188.0;

/// Hit-testable rectangle (logical px) of the reset button at the ribbon's
/// top-left. Tapping it returns the engine to its start state (SPEC §7).
pub fn reset_button_rect() -> Rect {
    Rect::new(14.0, 12.0, 42.0, 40.0)
}
/// Hit-testable rectangle of the collapse button beside reset. Tapping it
/// shrinks the keyboard to its edge handle.
pub fn collapse_button_rect() -> Rect {
    Rect::new(48.0, 12.0, 76.0, 40.0)
}
/// Hit-testable rectangle of the copy button beside collapse. Tapping it puts
/// all finalized text on the system clipboard.
pub fn copy_button_rect() -> Rect {
    Rect::new(82.0, 12.0, 110.0, 40.0)
}
/// Hit-testable rectangle of the context-capture button beside copy. Tapping
/// it takes the primary selection (clipboard as fallback) as the LLM's
/// auxiliary context block and shows a preview for verification (SPEC §6.8).
pub fn capture_button_rect() -> Rect {
    Rect::new(116.0, 12.0, 144.0, 40.0)
}
/// Hit-testable rectangle of the close button at the ribbon's top-right.
/// Tapping it quits keyb completely.
pub fn close_button_rect(lw: f64) -> Rect {
    Rect::new(lw - 42.0, 12.0, lw - 14.0, 40.0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RibbonControl {
    Reset,
    Collapse,
    Copy,
    CaptureContext,
    Close,
}

/// Resolve a ribbon click once so both window frontends have identical control
/// routing. The supported canvas width never allows these regions to overlap.
pub(crate) fn ribbon_control_at(lw: f64, point: Point) -> Option<RibbonControl> {
    if reset_button_rect().contains(point) {
        Some(RibbonControl::Reset)
    } else if collapse_button_rect().contains(point) {
        Some(RibbonControl::Collapse)
    } else if copy_button_rect().contains(point) {
        Some(RibbonControl::Copy)
    } else if capture_button_rect().contains(point) {
        Some(RibbonControl::CaptureContext)
    } else if close_button_rect(lw).contains(point) {
        Some(RibbonControl::Close)
    } else {
        None
    }
}

fn ribbon_text_rect(lw: f64) -> Rect {
    Rect::new(
        RIBBON_TEXT_X,
        0.0,
        (close_button_rect(lw).x0 - 2.0).max(RIBBON_TEXT_X),
        RIBBON_H,
    )
}

/// Extra gap (px) reserved between a box's label glyph and its nearest child's
/// left edge, so the glyph is never touched by the child's fill.
const LABEL_PAD: f64 = 4.0;

/// A box ready to draw: geometry, and its label (glyphs + size + baseline) if
/// tall enough. `x0` is filled in by the width recurrence.
struct BoxDraw {
    sym: u8,
    /// Set for group boxes (capitals/digits/symbols) — drives a distinct,
    /// popping color so the special boxes stand out from the letter palette.
    group: Option<GroupColor>,
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

/// Context-block UI handed to `build` each frame (SPEC §6.8): the chip while
/// a block is active, and the verification panel while it should be visible.
pub(crate) struct CtxView<'a> {
    /// Character count for the `ctx N` chip; `None` hides it.
    pub chip: Option<usize>,
    pub panel: Option<PanelView<'a>>,
}

pub(crate) struct PanelView<'a> {
    /// Header line, e.g. "context — 742 chars".
    pub title: String,
    /// The captured text (post byte-cap — what the model was promised), or
    /// the failure detail for an error panel.
    pub body: &'a str,
    pub error: bool,
}

/// Whether `point` hits the ctx chip drawn from the last frame's cache.
/// Left-click re-opens the preview; right-click clears the block.
pub(crate) fn context_chip_hit(ribbon: &RibbonCache, point: Point) -> bool {
    ribbon.chip_rect.is_some_and(|r| r.contains(point))
}

#[derive(Default)]
pub(crate) struct RibbonCache {
    committed: String,
    committed_glyphs: Vec<Glyph>,
    committed_advance: f32,
    pending: String,
    pending_glyphs: Vec<Glyph>,
    /// Hit rect of the ctx chip, refreshed every frame (`None` = no chip).
    chip_rect: Option<Rect>,
    /// Sanitized + wrapped panel lines, keyed by inputs so an open panel
    /// costs nothing per frame.
    panel_key: (String, usize, usize),
    panel_lines: Vec<String>,
    /// has_glyph answers mean font parses; memoize per character.
    glyph_cache: HashMap<char, bool>,
}

/// Characters the preview can show: newline splits lines, other control
/// chars become spaces, and anything the system font cannot render becomes
/// the replacement char (the model still receives the true bytes).
fn sanitize_panel_text(text: &str, mut has_glyph: impl FnMut(char) -> bool) -> String {
    text.chars()
        .map(|c| {
            if c == '\n' {
                '\n'
            } else if c.is_control() {
                ' '
            } else if has_glyph(c) {
                c
            } else {
                '\u{FFFD}'
            }
        })
        .collect()
}

/// Naive fixed-column wrap, then head+tail elision past `max_lines`. This is
/// a verification panel, not typesetting: mid-word breaks are fine.
fn wrap_panel_text(text: &str, max_chars: usize, max_lines: usize) -> Vec<String> {
    let max_chars = max_chars.max(1);
    let max_lines = max_lines.max(2);
    let mut lines = Vec::new();
    for raw in text.lines() {
        let mut rest = raw;
        loop {
            match rest.char_indices().nth(max_chars) {
                Some((cut, _)) => {
                    lines.push(rest[..cut].to_string());
                    rest = &rest[cut..];
                }
                None => {
                    lines.push(rest.to_string());
                    break;
                }
            }
        }
    }
    if lines.len() > max_lines {
        let head = max_lines * 2 / 3;
        let tail = max_lines - head - 1;
        let elided = lines.len() - head - tail;
        let mut out: Vec<String> = lines[..head].to_vec();
        out.push(format!("\u{2026} ({elided} lines elided) \u{2026}"));
        out.extend_from_slice(&lines[lines.len() - tail..]);
        out
    } else {
        lines
    }
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

#[allow(clippy::too_many_arguments)]
pub fn build(
    scene: &mut Scene,
    ribbon: &mut RibbonCache,
    engine: &Engine,
    fonts: &FontAssets,
    viewport: (f64, f64, f64),
    input: PointerInput,
    cursor: (f64, f64),
    ctx_view: CtxView<'_>,
) {
    let (lw, lh, scale) = viewport;
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
            let text = box_label_text(b.group, b.label, fonts);
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
                group: b.group,
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
        match d.group {
            // Group boxes read as *containers*, not writable leaves: a hollow
            // interior (faint category tint) ringed by a bright category-colored
            // border. Enclosed leaves draw on top, so the box reads as "a set of
            // these" rather than a single character you could type.
            Some(g) => {
                let c = group_color(g, d.depth, d.on_path);
                scene.fill(Fill::NonZero, t, c.with_alpha(0.16), None, &rect);
                let (stroke_c, stroke_w) = if d.on_path {
                    (Color::from_rgba8(0xFF, 0xFF, 0xFF, 190), 2.4)
                } else {
                    (c, 2.0)
                };
                scene.stroke(&Stroke::new(stroke_w), t, stroke_c, None, &rect);
            }
            None => {
                let fill_color = box_color(d.sym, d.depth, d.on_path);
                scene.fill(Fill::NonZero, t, fill_color, None, &rect);
                // Crosshair-path boxes (what will be written) get a bright edge.
                let (stroke_c, stroke_w) = if d.on_path {
                    (Color::from_rgba8(0xFF, 0xFF, 0xFF, 150), 1.6)
                } else {
                    (Color::from_rgba8(0, 0, 0, 60), 1.0)
                };
                scene.stroke(&Stroke::new(stroke_w), t, stroke_c, None, &rect);
            }
        }

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
    // While steering, trace the crosshair-to-cursor line first so the
    // crosshair tick draws on top of it.
    if input.engaged {
        scene.stroke(
            &Stroke::new(2.5),
            t,
            Color::from_rgba8(0xFF, 0x60, 0x60, 170),
            None,
            &Line::new((x_origin, cy), cursor),
        );
    }
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
    // Pending preview: text under the crosshair that will commit if the user
    // keeps steering here — shown dim past the caret so letters appear the
    // instant they are steered over, not only after the commit debounce.
    let pending: String = engine.pending_text().chars().take(24).collect();
    let text_clip = ribbon_text_rect(lw);
    let max_chars = (text_clip.width() / (font_px as f64 * 0.6)).max(4.0) as usize;
    let budget = max_chars.saturating_sub(pending.chars().count()).max(4);
    let shown = engine.committed_tail_text(budget).replace(' ', "\u{2423}");
    if ribbon.committed != shown {
        ribbon.committed = shown;
        (ribbon.committed_glyphs, ribbon.committed_advance) =
            fonts.layout(&ribbon.committed, font_px);
    }
    let shown_pending = pending.replace(' ', "\u{2423}");
    if ribbon.pending != shown_pending {
        ribbon.pending = shown_pending;
        (ribbon.pending_glyphs, _) = fonts.layout(&ribbon.pending, font_px);
    }
    let adv = ribbon.committed_advance;
    let baseline = RIBBON_H * 0.68;
    // Character-count budgeting is intentionally approximate, so clip shaped
    // text and its caret to guarantee they never paint under Close.
    scene.push_layer(Mix::Clip, 1.0, t, &text_clip);
    scene
        .draw_glyphs(&fonts.font)
        .font_size(font_px)
        .transform(t * Affine::translate((RIBBON_TEXT_X, baseline)))
        .brush(Color::from_rgb8(0xD8, 0xDC, 0xE8))
        .draw(Fill::NonZero, ribbon.committed_glyphs.iter().copied());
    scene.fill(
        Fill::NonZero,
        t,
        Color::from_rgb8(0x7A, 0xA2, 0xF7),
        None,
        &Rect::new(
            RIBBON_TEXT_X + adv as f64 + 3.0,
            baseline - font_px as f64 * 0.75,
            RIBBON_TEXT_X + adv as f64 + 6.0,
            baseline + font_px as f64 * 0.15,
        ),
    );
    if !ribbon.pending.is_empty() {
        scene
            .draw_glyphs(&fonts.font)
            .font_size(font_px)
            .transform(t * Affine::translate((RIBBON_TEXT_X + adv as f64 + 12.0, baseline)))
            .brush(Color::from_rgb8(0x6E, 0x76, 0x92))
            .draw(Fill::NonZero, ribbon.pending_glyphs.iter().copied());
    }
    scene.pop_layer();

    // Reset button at top-left: return to a fresh tree and empty context
    // without touching text already typed into the target app. Falls back to
    // `R` where the system font lacks the circular-arrow glyph.
    let btn = reset_button_rect();
    scene.fill(
        Fill::NonZero,
        t,
        Color::from_rgba8(0xFF, 0xFF, 0xFF, 24),
        None,
        &RoundedRect::from_rect(btn, 6.0),
    );
    let r_px = (btn.height() * 0.72) as f32;
    let label = if fonts.has_glyph('\u{21BA}') {
        "\u{21BA}"
    } else {
        "R"
    };
    let (glyphs, adv) = fonts.layout(label, r_px);
    scene
        .draw_glyphs(&fonts.font)
        .font_size(r_px)
        .transform(
            t * Affine::translate((
                btn.center().x - adv as f64 / 2.0,
                btn.center().y + r_px as f64 * 0.35,
            )),
        )
        .brush(Color::from_rgb8(0xD8, 0xDC, 0xE8))
        .draw(Fill::NonZero, glyphs.into_iter());

    // Collapse button beside reset. Since the keyboard is attached to the
    // right edge, the chevron points toward that edge and the collapsed tab.
    let btn = collapse_button_rect();
    scene.fill(
        Fill::NonZero,
        t,
        Color::from_rgba8(0xFF, 0xFF, 0xFF, 24),
        None,
        &RoundedRect::from_rect(btn, 6.0),
    );
    let collapse_px = (btn.height() * 0.72) as f32;
    let collapse_label = if fonts.has_glyph('\u{203A}') {
        "\u{203A}" // ›
    } else {
        ">"
    };
    let (glyphs, adv) = fonts.layout(collapse_label, collapse_px);
    scene
        .draw_glyphs(&fonts.font)
        .font_size(collapse_px)
        .transform(
            t * Affine::translate((
                btn.center().x - adv as f64 / 2.0,
                btn.center().y + collapse_px as f64 * 0.35,
            )),
        )
        .brush(Color::from_rgb8(0xD8, 0xDC, 0xE8))
        .draw(Fill::NonZero, glyphs.into_iter());

    // Copy button beside collapse: two offset page outlines avoid relying on
    // the system font having a recognizable clipboard glyph.
    let btn = copy_button_rect();
    scene.fill(
        Fill::NonZero,
        t,
        Color::from_rgba8(0xFF, 0xFF, 0xFF, 24),
        None,
        &RoundedRect::from_rect(btn, 6.0),
    );
    let back_page = RoundedRect::new(btn.x0 + 6.0, btn.y0 + 5.0, btn.x1 - 11.0, btn.y1 - 8.0, 2.0);
    let front_page = RoundedRect::new(btn.x0 + 11.0, btn.y0 + 9.0, btn.x1 - 6.0, btn.y1 - 4.0, 2.0);
    scene.stroke(
        &Stroke::new(1.6),
        t,
        Color::from_rgba8(0xD8, 0xDC, 0xE8, 160),
        None,
        &back_page,
    );
    scene.fill(
        Fill::NonZero,
        t,
        Color::from_rgb8(0x28, 0x2B, 0x39),
        None,
        &front_page,
    );
    scene.stroke(
        &Stroke::new(1.8),
        t,
        Color::from_rgb8(0xD8, 0xDC, 0xE8),
        None,
        &front_page,
    );

    // Context-capture button beside copy: ⧉ ("take the selection as
    // context"), with an ASCII stand-in where the system font lacks it.
    let btn = capture_button_rect();
    scene.fill(
        Fill::NonZero,
        t,
        Color::from_rgba8(0xFF, 0xFF, 0xFF, 24),
        None,
        &RoundedRect::from_rect(btn, 6.0),
    );
    let cap_px = (btn.height() * 0.72) as f32;
    let cap_label = if fonts.has_glyph('\u{29C9}') {
        "\u{29C9}"
    } else {
        "T"
    };
    let (glyphs, adv) = fonts.layout(cap_label, cap_px);
    scene
        .draw_glyphs(&fonts.font)
        .font_size(cap_px)
        .transform(
            t * Affine::translate((
                btn.center().x - adv as f64 / 2.0,
                btn.center().y + cap_px as f64 * 0.35,
            )),
        )
        .brush(Color::from_rgb8(0xD8, 0xDC, 0xE8))
        .draw(Fill::NonZero, glyphs.into_iter());

    // Close button at top-right. Unlike collapse, this exits keyb and removes
    // its native surface completely.
    let btn = close_button_rect(lw);
    scene.fill(
        Fill::NonZero,
        t,
        Color::from_rgba8(0xFF, 0xFF, 0xFF, 24),
        None,
        &RoundedRect::from_rect(btn, 6.0),
    );
    let close_px = (btn.height() * 0.72) as f32;
    let (glyphs, adv) = fonts.layout("\u{00D7}", close_px);
    scene
        .draw_glyphs(&fonts.font)
        .font_size(close_px)
        .transform(
            t * Affine::translate((
                btn.center().x - adv as f64 / 2.0,
                btn.center().y + close_px as f64 * 0.35,
            )),
        )
        .brush(Color::from_rgb8(0xD8, 0xDC, 0xE8))
        .draw(Fill::NonZero, glyphs.into_iter());

    // Context chip (SPEC §6.8): while a captured block is active, show its
    // size left of Close — lit while the preview panel is up. The rect is
    // cached so the frontends can hit-test clicks (left: pin the preview
    // open, right: clear the block).
    ribbon.chip_rect = None;
    if let Some(chars) = ctx_view.chip {
        let chip_px = 15.0f32;
        let label = format!("ctx {chars}");
        let (glyphs, adv) = fonts.layout(&label, chip_px);
        let w = adv as f64 + 16.0;
        let x1 = close_button_rect(lw).x0 - 8.0;
        let rect = Rect::new(x1 - w, 12.0, x1, 40.0);
        ribbon.chip_rect = Some(rect);
        let lit = ctx_view.panel.is_some();
        scene.fill(
            Fill::NonZero,
            t,
            Color::from_rgba8(0x7A, 0xA2, 0xF7, if lit { 110 } else { 40 }),
            None,
            &RoundedRect::from_rect(rect, 6.0),
        );
        scene
            .draw_glyphs(&fonts.font)
            .font_size(chip_px)
            .transform(
                t * Affine::translate((rect.x0 + 8.0, rect.center().y + chip_px as f64 * 0.35)),
            )
            .brush(Color::from_rgb8(0xD8, 0xDC, 0xE8))
            .draw(Fill::NonZero, glyphs.into_iter());
    }

    // Context preview panel (SPEC §6.8): the captured text, exactly as
    // promised to the model (post byte-cap), so the user can verify what
    // they are feeding it. Overlays the canvas but is never hit-testable —
    // steering clicks pass straight through to the engine.
    if let Some(panel) = &ctx_view.panel {
        let panel_px = 14.0f32;
        let line_h = 18.0f64;
        let x0 = 8.0;
        let x1 = lw - 8.0;
        let inner_w = (x1 - x0 - 24.0).max(40.0);
        let max_chars = (inner_w / (panel_px as f64 * 0.58)).max(8.0) as usize;
        let avail_h = (lh * 0.65 - RIBBON_H - 16.0).max(80.0);
        let max_lines = ((avail_h - 34.0) / line_h).clamp(4.0, 24.0) as usize;

        let key = (panel.body.to_string(), max_chars, max_lines);
        if ribbon.panel_key != key {
            let sanitized = sanitize_panel_text(panel.body, |c| {
                *ribbon
                    .glyph_cache
                    .entry(c)
                    .or_insert_with(|| fonts.has_glyph(c))
            });
            ribbon.panel_lines = wrap_panel_text(&sanitized, max_chars, max_lines);
            ribbon.panel_key = key;
        }

        let panel_top = RIBBON_H + 8.0;
        let panel_h = 10.0 + 20.0 + ribbon.panel_lines.len() as f64 * line_h + 10.0;
        let rect = RoundedRect::new(x0, panel_top, x1, panel_top + panel_h, 8.0);
        let (fill, edge) = if panel.error {
            (
                Color::from_rgba8(0x2B, 0x14, 0x14, 242),
                Color::from_rgba8(0xFF, 0x60, 0x60, 120),
            )
        } else {
            (
                Color::from_rgba8(0x0C, 0x0D, 0x14, 242),
                Color::from_rgba8(0xFF, 0xFF, 0xFF, 40),
            )
        };
        scene.fill(Fill::NonZero, t, fill, None, &rect);
        scene.stroke(&Stroke::new(1.0), t, edge, None, &rect);

        let (title_glyphs, _) = fonts.layout(&panel.title, panel_px);
        scene
            .draw_glyphs(&fonts.font)
            .font_size(panel_px)
            .transform(t * Affine::translate((x0 + 12.0, panel_top + 24.0)))
            .brush(if panel.error {
                Color::from_rgb8(0xFF, 0x90, 0x90)
            } else {
                Color::from_rgb8(0x7A, 0xA2, 0xF7)
            })
            .draw(Fill::NonZero, title_glyphs.into_iter());

        for (row, line) in ribbon.panel_lines.iter().enumerate() {
            let (glyphs, _) = fonts.layout(line, panel_px);
            scene
                .draw_glyphs(&fonts.font)
                .font_size(panel_px)
                .transform(
                    t * Affine::translate((
                        x0 + 12.0,
                        panel_top + 24.0 + 20.0 + row as f64 * line_h + panel_px as f64 * 0.35,
                    )),
                )
                .brush(Color::from_rgb8(0xB8, 0xBC, 0xC8))
                .draw(Fill::NonZero, glyphs.into_iter());
        }
    }

    // Speed readout, bottom-left.
    let (glyphs, _) = fonts.layout(
        &format!(
            "speed {:.1} bits per s   plus and minus adjust",
            engine.speed_bits
        ),
        13.0,
    );
    scene
        .draw_glyphs(&fonts.font)
        .font_size(13.0)
        .transform(t * Affine::translate((12.0, lh - 10.0)))
        .brush(Color::from_rgba8(0xFF, 0xFF, 0xFF, 110))
        .draw(Fill::NonZero, glyphs.into_iter());
}

/// Collapsed state: the small edge tab, with a centered chevron hinting "click
/// to open". No engine data needed.
///
/// The surface itself is opaque (vello presents with no alpha), so a rounded
/// tab would leave the dark clear color showing in the corners. Instead fill
/// the whole surface with the tab color.
pub fn build_handle(scene: &mut Scene, fonts: &FontAssets, lw: f64, lh: f64, scale: f64) {
    let t = Affine::scale(scale);
    scene.fill(
        Fill::NonZero,
        t,
        Color::from_rgb8(0x24, 0x28, 0x38),
        None,
        &Rect::new(0.0, 0.0, lw, lh),
    );
    let font_px = (lw * 0.7) as f32;
    let (glyphs, adv) = fonts.layout("\u{2039}", font_px); // ‹
    scene
        .draw_glyphs(&fonts.font)
        .font_size(font_px)
        .transform(
            t * Affine::translate((
                lw / 2.0 - adv as f64 / 2.0,
                lh / 2.0 + font_px as f64 * 0.35,
            )),
        )
        .brush(Color::from_rgb8(0xD8, 0xDC, 0xE8))
        .draw(Fill::NonZero, glyphs.into_iter());
}

/// Display mark for a box. Leaf boxes show their own character; group boxes get
/// a short, non-typeable mark so they're never mistaken for a letter you could
/// write (ISSUES.md): capitals a shift arrow (⇧), digits and symbols a bracketed
/// exemplar ("[1]", "[!]") whose brackets read as "a set of these". Falls back to
/// a bracketed letter where the system font lacks the shift glyph.
fn box_label_text(group: Option<GroupColor>, ch: char, fonts: &FontAssets) -> String {
    match group {
        Some(GroupColor::Capitals) if fonts.has_glyph('\u{21E7}') => "\u{21E7}".to_string(),
        Some(_) => format!("[{ch}]"),
        None => ch.to_string(),
    }
}

/// Golden-angle hue per symbol so neighbors in the column differ clearly;
/// depth alternates lightness (SPEC §7.1 palette placeholder). Boxes on the
/// crosshair path are lifted slightly so the pending text reads on canvas.
fn box_color(sym: u8, depth: usize, on_path: bool) -> Color {
    let hue = (sym as f64 * 137.508) % 360.0;
    let (s, mut l) = if depth % 2 == 0 {
        (0.42, 0.30)
    } else {
        (0.40, 0.38)
    };
    if on_path {
        l += 0.07;
    }
    hsl(hue, s, l)
}

/// Distinct, highly-saturated colors for the special group boxes so they pop
/// out of the muted per-letter palette (SPEC §5.4): capitals cyan, digits
/// amber, symbols magenta. Depth still nudges lightness so nesting reads, and
/// the crosshair path brightens as elsewhere.
fn group_color(group: GroupColor, depth: usize, on_path: bool) -> Color {
    let hue = match group {
        GroupColor::Capitals => 188.0, // cyan/teal
        GroupColor::Digits => 38.0,    // amber
        GroupColor::Symbols => 315.0,  // magenta
    };
    let s = 0.68;
    let mut l = if depth % 2 == 0 { 0.44 } else { 0.50 };
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
    use super::{
        box_left_edges, capture_button_rect, close_button_rect, collapse_button_rect,
        context_chip_hit, copy_button_rect, reset_button_rect, ribbon_control_at, ribbon_text_rect,
        sanitize_panel_text, wrap_panel_text, RibbonCache, RibbonControl, LABEL_PAD, MIN_CANVAS_W,
        RIBBON_TEXT_X,
    };
    use crate::RIBBON_H;
    use vello::kurbo::Point;

    const LW: f64 = 620.0;

    #[test]
    fn toolbar_buttons_are_aligned_and_separate() {
        for lw in [LW, MIN_CANVAS_W] {
            let reset = reset_button_rect();
            let collapse = collapse_button_rect();
            let copy = copy_button_rect();
            let capture = capture_button_rect();
            let close = close_button_rect(lw);
            for button in [reset, collapse, copy, capture, close] {
                assert!(button.x0 >= 0.0 && button.x1 <= lw, "within canvas width");
                assert!(
                    button.y0 >= 0.0 && button.y1 <= RIBBON_H,
                    "within the ribbon"
                );
            }
            assert!(reset.x1 < collapse.x0, "left controls do not overlap");
            assert!(collapse.x1 < copy.x0, "copy is separate from collapse");
            assert!(copy.x1 < capture.x0, "capture is separate from copy");
            assert!(capture.x1 < close.x0, "close is separate on the right");
            assert!(reset.x0 < lw * 0.2, "reset is anchored to the left");
            assert!(close.x1 > lw * 0.8, "close is anchored to the right");
        }
        assert!(
            capture_button_rect().x1 < RIBBON_TEXT_X,
            "left controls do not overlap ribbon text"
        );
        assert!(
            ribbon_text_rect(LW).x1 <= close_button_rect(LW).x0,
            "ribbon text is clipped before close"
        );
    }

    #[test]
    fn ribbon_clicks_route_to_distinct_controls() {
        for (button, expected) in [
            (reset_button_rect(), RibbonControl::Reset),
            (collapse_button_rect(), RibbonControl::Collapse),
            (copy_button_rect(), RibbonControl::Copy),
            (capture_button_rect(), RibbonControl::CaptureContext),
            (close_button_rect(MIN_CANVAS_W), RibbonControl::Close),
        ] {
            assert_eq!(
                ribbon_control_at(MIN_CANVAS_W, button.center()),
                Some(expected)
            );
        }
        assert_eq!(
            ribbon_control_at(MIN_CANVAS_W, Point::new(100.0, 50.0)),
            None
        );
    }

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
        assert!(
            (x0[0] - (LW - 200.0 * 1.1)).abs() < 1e-6,
            "parent stays nominal"
        );
    }

    #[test]
    fn chip_hit_testing_follows_cached_rect() {
        let mut ribbon = RibbonCache::default();
        assert!(!context_chip_hit(&ribbon, Point::new(500.0, 26.0)));
        ribbon.chip_rect = Some(vello::kurbo::Rect::new(480.0, 12.0, 520.0, 40.0));
        assert!(context_chip_hit(&ribbon, Point::new(500.0, 26.0)));
        assert!(!context_chip_hit(&ribbon, Point::new(500.0, 50.0)));
        assert!(!context_chip_hit(&ribbon, Point::new(100.0, 26.0)));
    }

    #[test]
    fn panel_sanitization_keeps_only_renderable_text() {
        // Stub font: ASCII letters and space only.
        let has = |c: char| c.is_ascii_lowercase() || c == ' ';
        let out = sanitize_panel_text("hi\tthere\n\u{1F600}x\u{7F}", has);
        assert_eq!(out, "hi there\n\u{FFFD}x ");
    }

    #[test]
    fn panel_wrap_breaks_long_lines_and_keeps_empty_ones() {
        let lines = wrap_panel_text("abcdef\n\ngh", 3, 10);
        assert_eq!(lines, vec!["abc", "def", "", "gh"]);
    }

    #[test]
    fn panel_wrap_elides_the_middle_of_long_captures() {
        let text: String = (0..30).map(|i| format!("line {i:02}\n")).collect();
        let lines = wrap_panel_text(&text, 80, 9);
        assert_eq!(lines.len(), 9);
        assert_eq!(lines[0], "line 00");
        assert!(
            lines[6].contains("lines elided"),
            "elision marker: {:?}",
            lines[6]
        );
        assert_eq!(lines[8], "line 29", "tail survives: {lines:?}");
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

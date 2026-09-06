//! keyb-ui: the zooming canvas (SPEC §7).
//!
//! Two window frontends share one frame core:
//! - `layer_shell`: Wayland wlr-layer-shell overlay that can never take
//!   keyboard focus (SPEC §7.3) — the OSK proper.
//! - `winit_shell`: X11 override-redirect/dock OSK, and a plain window for
//!   development.
//!
//! Committed-text changes are forwarded to `sink` every frame; keyb-app
//! routes them to the injection worker (keyb-ui never depends on
//! keyb-inject).

mod font;
mod layer_shell;
mod niri_focus;
mod scene;
mod winit_shell;

use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use keyb_core::{Engine, PointerInput, TextEvent};
use vello::peniko::Color;
use vello::util::{RenderContext, RenderSurface};
use vello::{AaConfig, AaSupport, RenderParams, Renderer, RendererOptions, Scene};

/// Height of the committed-text ribbon (logical px).
pub const RIBBON_H: f64 = 56.0;
/// Origin line: pointer right of this fraction of the canvas zooms in,
/// left of it zooms out.
pub const ORIGIN_X_FRAC: f64 = 0.18;
/// Fallback width of the OSK column (logical px) for the layer-shell and X11
/// dock frontends, used when the output size is unknown. `KEYB_WIDTH`
/// overrides; otherwise the column is ~1/3 of the output width. The column is
/// always full height.
pub const OSK_W: u32 = 620;
/// Size of the collapsed handle/tab (logical px) shown on the screen edge when
/// the keyboard is collapsed. Clicking it brings the OSK back.
pub const HANDLE_W: u32 = 28;
pub const HANDLE_H: u32 = 110;

/// Byte cap for a captured context block (SPEC §6.8); the tail is kept, so
/// the most recent text always survives. The LLM worker applies its own
/// tighter token budget on top.
pub const CONTEXT_MAX_BYTES: usize = 16 * 1024;
/// How long the context preview panel stays up after a capture.
const CONTEXT_PREVIEW_SECS: f64 = 6.0;
/// How long a capture-failure message stays up.
const CONTEXT_ERROR_SECS: f64 = 5.0;

/// The captured context block (SPEC §6.8) as the UI promised it to the model.
pub struct ContextState {
    /// Post-cap text handed to `Engine::set_context`.
    text: String,
    /// Display count, computed once at capture.
    chars: usize,
    /// The raw capture exceeded the byte cap; the panel notes the trim.
    trimmed: bool,
    /// Transient post-capture confirmation deadline.
    preview_until: Instant,
    /// Chip click pins the panel open; another click dismisses it.
    pinned: bool,
}

/// Receives committed-text changes once per frame, on the UI thread.
pub type TextSink = Box<dyn FnMut(TextEvent)>;

/// Width of the OSK column in logical px. `KEYB_WIDTH` (absolute px) wins;
/// otherwise 1/3 of the output width when known, falling back to `OSK_W`.
pub(crate) fn osk_width(output_w: Option<u32>) -> u32 {
    std::env::var("KEYB_WIDTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .or_else(|| output_w.map(|w| w / 3))
        .unwrap_or(OSK_W)
        .clamp(240, 3000)
}

/// Frame logic shared by both frontends: pointer mapping, engine stepping,
/// event draining, scene building.
pub(crate) struct Core {
    pub engine: Engine,
    pub fonts: font::FontAssets,
    pub scene: Scene,
    /// Hit-test cache for the ctx chip (SPEC §6.8); frontends read it.
    pub(crate) ribbon: scene::RibbonCache,
    pub cursor: (f64, f64),
    pub engaged: bool,
    /// Whether the OSK is expanded. When false the engine is frozen and only
    /// the collapsed handle is drawn (SPEC §7.3: dismissible overlay).
    pub visible: bool,
    pub sink: TextSink,
    /// Lazily opened so startup still succeeds when clipboard access is
    /// unavailable. A successful handle is retained to preserve X11 ownership.
    clipboard: Option<arboard::Clipboard>,
    /// Active context block (SPEC §6.8), shown as a ribbon chip.
    context: Option<ContextState>,
    /// Transient capture-failure message shown in place of the preview.
    flash: Option<(String, Instant)>,
    last_frame: Instant,
}

impl Core {
    pub fn new(engine: Engine, sink: TextSink) -> Result<Self> {
        Ok(Core {
            engine,
            fonts: font::FontAssets::load_system()?,
            scene: Scene::new(),
            ribbon: scene::RibbonCache::default(),
            cursor: (0.0, 0.0),
            engaged: false,
            visible: true,
            sink,
            clipboard: None,
            context: None,
            flash: None,
            last_frame: Instant::now(),
        })
    }

    pub fn adjust_speed(&mut self, delta: f64) {
        // Under auto speed control the governor re-derives `speed_bits` every
        // frame, so a nudge must move its cap or it would be overridden on
        // the next step.
        if self.engine.params.adaptive_speed {
            self.engine.params.max_bits_per_sec =
                (self.engine.params.max_bits_per_sec + delta).clamp(1.0, 10.0);
        }
        self.engine.speed_bits = (self.engine.speed_bits + delta).clamp(1.0, 10.0);
    }

    /// Reset the engine to its start state and drop any in-progress steering.
    /// Text already injected into the target app is left untouched (SPEC §7).
    /// The captured context block goes too: `Engine::reset` clears it on the
    /// LM side (SPEC §6.8), and the UI drops its display copy here.
    pub fn reset(&mut self) {
        self.engine.reset();
        self.context = None;
        self.flash = None;
        self.engaged = false;
    }

    /// Capture the primary selection (falling back to the clipboard) as the
    /// LLM's auxiliary context block (SPEC §6.8), and show the preview panel
    /// so the user can verify what the model will see.
    pub fn capture_context(&mut self) {
        match self.read_selection() {
            Ok(raw) => {
                let capped = cap_tail(&raw, CONTEXT_MAX_BYTES).to_string();
                self.engine.set_context(Some(&capped));
                dump_context(&capped);
                self.flash = None;
                self.context = Some(ContextState {
                    chars: capped.chars().count(),
                    trimmed: capped.len() < raw.len(),
                    text: capped,
                    preview_until: Instant::now() + Duration::from_secs_f64(CONTEXT_PREVIEW_SECS),
                    pinned: false,
                });
            }
            Err(error) => {
                self.flash = Some((
                    format!("{error:#}"),
                    Instant::now() + Duration::from_secs_f64(CONTEXT_ERROR_SECS),
                ));
            }
        }
    }

    /// Left-click on the ctx chip: pin the preview panel open or dismiss it.
    pub fn toggle_context_preview(&mut self) {
        if let Some(c) = &mut self.context {
            c.pinned = !c.pinned;
        }
    }

    /// Right-click on the ctx chip: drop the context block without touching
    /// zoom state or committed text.
    pub fn clear_context(&mut self) {
        if self.context.take().is_some() {
            self.engine.set_context(None);
        }
    }

    /// Read the text the user most likely means as context: the primary
    /// selection (mark text, click ⧉ — no copying step), falling back to the
    /// regular clipboard for compositors without primary selection and for
    /// Ctrl+C habits.
    fn read_selection(&mut self) -> Result<String> {
        if self.clipboard.is_none() {
            self.clipboard = Some(arboard::Clipboard::new().context("open system clipboard")?);
        }
        let result = read_selection_text(self.clipboard.as_mut().expect("initialized above"));
        if result.is_err() {
            // Reconnect on the next click rather than trusting a connection
            // that just failed (same discipline as copy_committed_text).
            self.clipboard = None;
        }
        result
    }

    /// Copy exactly the finalized text shown before the ribbon caret. Pending
    /// crosshair preview characters are deliberately excluded.
    pub fn copy_committed_text(&mut self) -> Result<()> {
        let text = self.engine.committed_text();
        if self.clipboard.is_none() {
            self.clipboard = Some(arboard::Clipboard::new().context("open system clipboard")?);
        }
        let result = self
            .clipboard
            .as_mut()
            .expect("clipboard was initialized above")
            .set_text(text);
        match result {
            Ok(()) => Ok(()),
            Err(error) => {
                // Reconnect on the next click rather than caching a stale X11
                // or Wayland connection after a transient failure.
                self.clipboard = None;
                Err(error).context("write committed text to system clipboard")
            }
        }
    }

    fn pointer_input(&self, width: f64, height: f64) -> PointerInput {
        let (px, py) = self.cursor;
        let canvas_h = (height - RIBBON_H).max(1.0);
        let x_origin = width * ORIGIN_X_FRAC;
        let dx = if px >= x_origin {
            (px - x_origin) / (width - x_origin).max(1.0)
        } else {
            (px - x_origin) / x_origin.max(1.0)
        };
        // Mild expansion around the center so edge targets stay reachable.
        let y_raw = (py - RIBBON_H) / canvas_h;
        let y = 0.5 + (y_raw - 0.5) * 1.15;
        PointerInput {
            dx: dx.clamp(-1.0, 1.0),
            y: y.clamp(-0.15, 1.15),
            engaged: self.engaged,
        }
    }

    /// Step the engine and rebuild the scene for a `lw` x `lh` logical-px
    /// canvas at `scale`. Committed-text changes flow into the sink here.
    pub fn tick(&mut self, lw: f64, lh: f64, scale: f64) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f64().clamp(0.0, 0.05);
        self.last_frame = now;

        self.scene.reset();
        if !self.visible {
            // Frozen: no engine step, no commits — just the collapsed handle.
            scene::build_handle(&mut self.scene, &self.fonts, lw, lh, scale);
            return;
        }

        let input = self.pointer_input(lw, lh);
        self.engine.set_viewport_px(lh - RIBBON_H);
        self.engine.step(input, dt);
        for ev in self.engine.drain_text_events() {
            (self.sink)(ev);
        }

        let ctx_view = context_view(&self.context, &self.flash, self.engaged);
        scene::build(
            &mut self.scene,
            &mut self.ribbon,
            &self.engine,
            &self.fonts,
            (lw, lh, scale),
            input,
            self.cursor,
            ctx_view,
        );
    }
}

/// What the scene needs to draw the context UI this frame: the chip while
/// a block is active, plus the panel while transiently or pinned visible.
/// The transient confirmation hides while steering; the pinned panel and
/// error flash stay. A free function over disjoint fields, so `tick` can
/// borrow the scene and ribbon mutably at the same time.
fn context_view<'a>(
    context: &'a Option<ContextState>,
    flash: &'a Option<(String, Instant)>,
    engaged: bool,
) -> scene::CtxView<'a> {
    let now = Instant::now();
    if let Some((msg, until)) = flash {
        if now < *until {
            return scene::CtxView {
                chip: context.as_ref().map(|c| c.chars),
                panel: Some(scene::PanelView {
                    title: "context capture failed".into(),
                    body: msg.as_str(),
                    error: true,
                }),
            };
        }
    }
    let panel = context.as_ref().and_then(|c| {
        let transient = now < c.preview_until && !engaged;
        (c.pinned || transient).then(|| scene::PanelView {
            title: if c.trimmed {
                format!("context — {} chars (tail of a longer capture)", c.chars)
            } else {
                format!("context — {} chars", c.chars)
            },
            body: c.text.as_str(),
            error: false,
        })
    });
    scene::CtxView {
        chip: context.as_ref().map(|c| c.chars),
        panel,
    }
}

/// Keep the last `max` bytes of `text`, on a char boundary: recent context
/// matters more than the beginning of a capture.
fn cap_tail(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut start = text.len() - max;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

fn read_selection_text(cb: &mut arboard::Clipboard) -> Result<String> {
    use arboard::{GetExtLinux, LinuxClipboardKind};
    let primary = cb.get().clipboard(LinuxClipboardKind::Primary).text();
    let text = match primary {
        Ok(t) if !t.trim().is_empty() => t,
        _ => {
            let t = cb.get_text().context("read selection or clipboard")?;
            if t.trim().is_empty() {
                anyhow::bail!("nothing selected, and the clipboard is empty");
            }
            t
        }
    };
    Ok(text)
}

/// Debug/verification dump of the exact bytes promised to the model:
/// `$XDG_STATE_HOME/keyb/last_context.txt` (SPEC §6.8). A few KB written on
/// a deliberate click — infrequent enough to stay on the UI thread.
fn dump_context(text: &str) {
    let dir = std::env::var_os("XDG_STATE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".local/state"))
        })
        .map(|d| d.join("keyb"));
    let Some(dir) = dir else { return };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let _ = std::fs::write(dir.join("last_context.txt"), text);
}

/// One vello render context + per-device renderers, shared by frontends.
pub(crate) struct Gpu {
    pub context: RenderContext,
    /// Renderer pipelines are surface-format-specific. A migrated layer may
    /// keep the same device while selecting a different format.
    renderers: Vec<Option<(vello::wgpu::TextureFormat, Renderer)>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RenderOutcome {
    Presented,
    Retry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SurfaceErrorAction {
    Retry,
    Reconfigure,
    Fatal,
}

fn surface_error_action(error: &vello::wgpu::SurfaceError) -> SurfaceErrorAction {
    match error {
        vello::wgpu::SurfaceError::Timeout => SurfaceErrorAction::Retry,
        vello::wgpu::SurfaceError::Outdated | vello::wgpu::SurfaceError::Lost => {
            SurfaceErrorAction::Reconfigure
        }
        vello::wgpu::SurfaceError::OutOfMemory => SurfaceErrorAction::Fatal,
    }
}

impl Gpu {
    pub fn new() -> Self {
        Gpu {
            context: RenderContext::new(),
            renderers: Vec::new(),
        }
    }

    pub fn ensure_renderer(
        &mut self,
        dev_id: usize,
        format: vello::wgpu::TextureFormat,
    ) -> Result<()> {
        self.renderers
            .resize_with(self.context.devices.len().max(dev_id + 1), || None);
        let format_changed = self.renderers[dev_id]
            .as_ref()
            .is_some_and(|(configured_format, _)| *configured_format != format);
        if self.renderers[dev_id].is_none() || format_changed {
            let renderer = Renderer::new(
                &self.context.devices[dev_id].device,
                RendererOptions {
                    surface_format: Some(format),
                    use_cpu: false,
                    antialiasing_support: AaSupport::area_only(),
                    num_init_threads: None,
                },
            )
            .context("create vello renderer")?;
            self.renderers[dev_id] = Some((format, renderer));
        }
        Ok(())
    }

    /// Render `scene` to the current surface texture. Transient acquisition
    /// failures are returned as `Retry`; lost/outdated surfaces are
    /// reconfigured first. Pipeline and memory failures remain explicit.
    pub fn render(
        &mut self,
        surface: &mut RenderSurface<'_>,
        scene: &Scene,
        w: u32,
        h: u32,
    ) -> Result<RenderOutcome> {
        let surface_texture = match surface.surface.get_current_texture() {
            Ok(t) => t,
            Err(error) => match surface_error_action(&error) {
                SurfaceErrorAction::Retry => return Ok(RenderOutcome::Retry),
                SurfaceErrorAction::Reconfigure => {
                    self.context.resize_surface(surface, w.max(1), h.max(1));
                    return Ok(RenderOutcome::Retry);
                }
                SurfaceErrorAction::Fatal => {
                    return Err(anyhow::Error::new(error).context("acquire render surface"));
                }
            },
        };
        let device_handle = &self.context.devices[surface.dev_id];
        self.renderers
            .get_mut(surface.dev_id)
            .and_then(Option::as_mut)
            .map(|(_, renderer)| renderer)
            .context("renderer was not initialized for the surface device")?
            .render_to_surface(
                &device_handle.device,
                &device_handle.queue,
                scene,
                &surface_texture,
                &RenderParams {
                    base_color: Color::from_rgb8(0x14, 0x15, 0x20),
                    width: w,
                    height: h,
                    antialiasing_method: AaConfig::Area,
                },
            )
            .context("render vello scene")?;
        surface_texture.present();
        Ok(RenderOutcome::Presented)
    }
}

#[cfg(test)]
mod context_tests {
    use super::{cap_tail, context_view, ContextState, CONTEXT_MAX_BYTES};
    use std::time::{Duration, Instant};

    #[test]
    fn cap_tail_keeps_recent_bytes_on_a_char_boundary() {
        assert_eq!(cap_tail("hello", 16), "hello");
        assert_eq!(cap_tail("abcdefgh", 8), "abcdefgh");
        assert_eq!(cap_tail("abcdefgh", 4), "efgh");
        // "aé" is 1 + 2 bytes; max=2 lands on the start of é.
        assert_eq!(cap_tail("aé", 2), "é");
        // max=1 cannot fit é, so the slice is empty rather than a partial char.
        assert_eq!(cap_tail("aé", 1), "");
        let long = "x".repeat(CONTEXT_MAX_BYTES + 10);
        let kept = cap_tail(&long, CONTEXT_MAX_BYTES);
        assert_eq!(kept.len(), CONTEXT_MAX_BYTES);
        assert!(long.ends_with(kept));
    }

    fn sample(text: &str, trimmed: bool, preview_until: Instant, pinned: bool) -> ContextState {
        ContextState {
            chars: text.chars().count(),
            trimmed,
            text: text.to_string(),
            preview_until,
            pinned,
        }
    }

    #[test]
    fn context_view_hides_everything_when_idle() {
        let v = context_view(&None, &None, false);
        assert!(v.chip.is_none());
        assert!(v.panel.is_none());
    }

    #[test]
    fn context_view_flash_wins_while_live() {
        let context = Some(sample(
            "hello",
            false,
            Instant::now() + Duration::from_secs(10),
            false,
        ));
        let flash = Some(("boom".into(), Instant::now() + Duration::from_secs(5)));
        let v = context_view(&context, &flash, false);
        assert_eq!(v.chip, Some(5));
        let panel = v.panel.expect("error panel");
        assert!(panel.error);
        assert_eq!(panel.body, "boom");
        assert!(panel.title.contains("failed"));
    }

    #[test]
    fn context_view_expired_flash_falls_through_to_preview() {
        let context = Some(sample(
            "hello",
            false,
            Instant::now() + Duration::from_secs(10),
            false,
        ));
        let flash = Some(("boom".into(), Instant::now() - Duration::from_secs(1)));
        let v = context_view(&context, &flash, false);
        let panel = v.panel.expect("preview");
        assert!(!panel.error);
        assert_eq!(panel.body, "hello");
        assert!(panel.title.contains("5 chars"));
    }

    #[test]
    fn context_view_transient_hides_while_steering() {
        let context = Some(sample(
            "hello",
            false,
            Instant::now() + Duration::from_secs(10),
            false,
        ));
        let v = context_view(&context, &None, true);
        assert_eq!(v.chip, Some(5));
        assert!(v.panel.is_none(), "steering hides the confirmation");
    }

    #[test]
    fn context_view_pinned_survives_steering_and_expiry() {
        let context = Some(sample(
            "hello",
            true,
            Instant::now() - Duration::from_secs(1),
            true,
        ));
        let v = context_view(&context, &None, true);
        assert_eq!(v.chip, Some(5));
        let panel = v.panel.expect("pinned");
        assert!(!panel.error);
        assert!(panel.title.contains("tail"));
        assert_eq!(panel.body, "hello");
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::{surface_error_action, SurfaceErrorAction};
    use vello::wgpu::SurfaceError;

    #[test]
    fn surface_errors_have_explicit_recovery_actions() {
        assert_eq!(
            surface_error_action(&SurfaceError::Timeout),
            SurfaceErrorAction::Retry
        );
        assert_eq!(
            surface_error_action(&SurfaceError::Outdated),
            SurfaceErrorAction::Reconfigure
        );
        assert_eq!(
            surface_error_action(&SurfaceError::Lost),
            SurfaceErrorAction::Reconfigure
        );
        assert_eq!(
            surface_error_action(&SurfaceError::OutOfMemory),
            SurfaceErrorAction::Fatal
        );
    }
}

/// Which window frontend to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shell {
    /// Wayland layer-shell overlay (never takes keyboard focus).
    Layer,
    /// X11 override-redirect dock strip.
    X11Osk,
    /// Ordinary window (development).
    Window,
}

fn pick_shell() -> Shell {
    match std::env::var("KEYB_SHELL").ok().as_deref() {
        Some("layer") => Shell::Layer,
        Some("x11") => Shell::X11Osk,
        Some("window") => Shell::Window,
        Some(other) => {
            eprintln!("keyb: unknown KEYB_SHELL={other}, using auto");
            pick_shell_auto()
        }
        None => pick_shell_auto(),
    }
}

fn pick_shell_auto() -> Shell {
    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        Shell::Layer
    } else if std::env::var_os("DISPLAY").is_some() {
        Shell::X11Osk
    } else {
        Shell::Window
    }
}

/// Run the UI until it exits. Takes ownership of the engine; `sink` receives
/// every committed-text change (SPEC §8).
pub fn run(engine: Engine, sink: TextSink) -> Result<()> {
    let core = Core::new(engine, sink)?;
    match pick_shell() {
        Shell::Layer => match layer_shell::run(core) {
            Ok(()) => Ok(()),
            Err(boxed) => {
                let (core, e) = *boxed;
                eprintln!("keyb: layer shell failed ({e:#}); falling back to a window");
                winit_shell::run(core, false)
            }
        },
        Shell::X11Osk => winit_shell::run(core, true),
        Shell::Window => winit_shell::run(core, false),
    }
}

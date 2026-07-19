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
mod scene;
mod winit_shell;

use std::time::Instant;

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
/// the keyboard is hidden. Clicking it (or a SIGUSR1) brings the OSK back.
pub const HANDLE_W: u32 = 28;
pub const HANDLE_H: u32 = 110;

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
    ribbon: scene::RibbonCache,
    pub cursor: (f64, f64),
    pub engaged: bool,
    /// Whether the OSK is shown. When false the engine is frozen and only the
    /// collapsed handle is drawn (SPEC §7.3: dismissible overlay).
    pub visible: bool,
    pub sink: TextSink,
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
            last_frame: Instant::now(),
        })
    }

    pub fn adjust_speed(&mut self, delta: f64) {
        self.engine.speed_bits = (self.engine.speed_bits + delta).clamp(1.0, 10.0);
    }

    /// Reset the engine to its start state and drop any in-progress steering.
    /// Text already injected into the target app is left untouched (SPEC §7).
    pub fn reset(&mut self) {
        self.engine.reset();
        self.engaged = false;
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

        scene::build(
            &mut self.scene,
            &mut self.ribbon,
            &self.engine,
            &self.fonts,
            (lw, lh, scale),
            input,
            self.cursor,
        );
    }
}

/// One vello render context + per-device renderers, shared by frontends.
pub(crate) struct Gpu {
    pub context: RenderContext,
    renderers: Vec<Option<Renderer>>,
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
        if self.renderers[dev_id].is_none() {
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
            self.renderers[dev_id] = Some(renderer);
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

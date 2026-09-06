//! winit frontend: a plain window for development, or an X11
//! override-redirect dock column for OSK use (SPEC §7.3).
//!
//! Override-redirect makes the window unmanaged: the WM never gives it
//! keyboard focus, so clicks steer keyb while keystrokes keep flowing to the
//! target app — the X11 analogue of layer-shell `keyboard_interactivity:
//! none`. `_NET_WM_WINDOW_TYPE_DOCK` keeps compositing WMs from shadowing or
//! animating it.

use std::sync::Arc;

use anyhow::{Context as _, Result};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::scene::{context_chip_hit, ribbon_control_at, RibbonControl, MIN_CANVAS_W};
use crate::{osk_width, Core, Gpu, RIBBON_H};

struct RenderState {
    surface: vello::util::RenderSurface<'static>,
    window: Arc<Window>,
}

fn clamp_osk_width(width: u32, monitor_width: u32, scale: f64) -> u32 {
    let min_width = ((MIN_CANVAS_W * scale).ceil() as u32).min(monitor_width);
    width.clamp(min_width, monitor_width)
}

struct App {
    core: Core,
    gpu: Gpu,
    state: Option<RenderState>,
    error: Option<anyhow::Error>,
    /// Borderless bottom-anchored override-redirect strip instead of a
    /// normal window.
    osk: bool,
}

impl App {
    fn frame(&mut self) -> Result<()> {
        let Some(state) = &mut self.state else {
            return Ok(());
        };
        let size = state.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        let scale = state.window.scale_factor();
        self.core
            .tick(size.width as f64 / scale, size.height as f64 / scale, scale);
        self.gpu.render(
            &mut state.surface,
            &self.core.scene,
            size.width,
            size.height,
        )?;
        Ok(())
    }

    fn resume(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let window = Arc::new(
            event_loop
                .create_window(self.window_attributes(event_loop))
                .context("create window")?,
        );
        let size = window.inner_size();
        let surface = pollster::block_on(self.gpu.context.create_surface(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            vello::wgpu::PresentMode::AutoVsync,
        ))
        .context("create render surface")?;
        self.gpu
            .ensure_renderer(surface.dev_id, surface.format)
            .context("initialize renderer")?;
        self.state = Some(RenderState { surface, window });
        Ok(())
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, error: anyhow::Error) {
        self.core.engaged = false;
        self.error = Some(error);
        event_loop.exit();
    }

    /// Left-button press: activate a ribbon control / edge handle, otherwise
    /// engage steering (mirrors the layer-shell frontend).
    fn press(&mut self, event_loop: &ActiveEventLoop) {
        let Some(state) = &self.state else { return };
        let scale = state.window.scale_factor();
        let lw = state.window.inner_size().width as f64 / scale;
        let (cx, cy) = self.core.cursor;
        if !self.core.visible {
            self.core.visible = true;
        } else if context_chip_hit(&self.core.ribbon, vello::kurbo::Point::new(cx, cy)) {
            self.core.toggle_context_preview();
        } else {
            match ribbon_control_at(lw, vello::kurbo::Point::new(cx, cy)) {
                Some(RibbonControl::Collapse) => {
                    self.core.visible = false;
                    self.core.engaged = false;
                }
                Some(RibbonControl::Reset) => self.core.reset(),
                Some(RibbonControl::Copy) => {
                    self.core.engaged = false;
                    if let Err(error) = self.core.copy_committed_text() {
                        eprintln!("keyb: could not copy text: {error:#}");
                    }
                }
                Some(RibbonControl::CaptureContext) => {
                    self.core.engaged = false;
                    self.core.capture_context();
                }
                Some(RibbonControl::Close) => {
                    self.core.engaged = false;
                    event_loop.exit();
                }
                None => self.core.engaged = true,
            }
        }
    }

    /// Right-button press: clears the context block when the ctx chip is hit
    /// (SPEC §6.8); anywhere else it does nothing.
    fn right_press(&mut self) {
        let (cx, cy) = self.core.cursor;
        if context_chip_hit(&self.core.ribbon, vello::kurbo::Point::new(cx, cy)) {
            self.core.clear_context();
        }
    }

    fn window_attributes(&self, event_loop: &ActiveEventLoop) -> winit::window::WindowAttributes {
        let attrs = Window::default_attributes().with_title("keyb");
        if !self.osk {
            return attrs
                .with_inner_size(LogicalSize::new(1200.0, 800.0))
                .with_min_inner_size(LogicalSize::new(MIN_CANVAS_W, RIBBON_H));
        }
        // Full-height column on the right edge of the primary monitor.
        let monitor = event_loop
            .primary_monitor()
            .or_else(|| event_loop.available_monitors().next());
        let (msize, mpos, mscale) = monitor
            .map(|m| (m.size(), m.position(), m.scale_factor()))
            .unwrap_or((
                PhysicalSize::new(1920, 1080),
                PhysicalPosition::new(0, 0),
                1.0,
            ));
        // KEYB_WIDTH is logical px; otherwise 1/3 of the monitor. Widths here
        // are physical, so scale the env override and pass the physical monitor
        // width to `osk_width` for the 1/3 default.
        let w = std::env::var("KEYB_WIDTH")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .map(|lw| (lw * mscale) as u32)
            .unwrap_or_else(|| osk_width(Some(msize.width)));
        // Keep all four ribbon controls independently clickable even when a
        // smaller KEYB_WIDTH was requested.
        let w = clamp_osk_width(w, msize.width, mscale);
        let attrs = attrs
            .with_inner_size(PhysicalSize::new(w, msize.height))
            .with_position(PhysicalPosition::new(
                mpos.x + msize.width.saturating_sub(w) as i32,
                mpos.y,
            ))
            .with_decorations(false)
            .with_resizable(false);
        #[cfg(all(unix, not(target_os = "macos")))]
        let attrs = {
            use winit::platform::x11::{WindowAttributesExtX11, WindowType};
            attrs
                .with_override_redirect(true)
                .with_x11_window_type(vec![WindowType::Dock])
        };
        attrs
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() || self.error.is_some() {
            return;
        }
        if let Err(error) = self.resume(event_loop) {
            self.fail(event_loop, error);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    self.gpu.context.resize_surface(
                        &mut state.surface,
                        size.width.max(1),
                        size.height.max(1),
                    );
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(state) = &self.state {
                    let scale = state.window.scale_factor();
                    self.core.cursor = (position.x / scale, position.y / scale);
                }
            }
            WindowEvent::CursorLeft { .. } => {
                // No pointer capture yet: a drag that escapes the window
                // would otherwise stay engaged forever.
                self.core.engaged = false;
            }
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Left => {
                    if state == ElementState::Pressed {
                        self.press(event_loop);
                    } else {
                        self.core.engaged = false;
                    }
                }
                MouseButton::Right if state == ElementState::Pressed => self.right_press(),
                _ => {}
            },
            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y / 40.0,
                };
                self.core.adjust_speed(0.25 * lines);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => event_loop.exit(),
                        Key::Character(ref c) if c == "+" || c == "=" => {
                            self.core.adjust_speed(0.5);
                        }
                        Key::Character(ref c) if c == "-" => {
                            self.core.adjust_speed(-0.5);
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.frame() {
                    self.fail(event_loop, error);
                    return;
                }
                if let Some(state) = &self.state {
                    state.window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

/// Run under winit until the window closes. `osk` selects the
/// override-redirect dock strip (X11 targets).
pub fn run(core: Core, osk: bool) -> Result<()> {
    let event_loop = if osk {
        // The OSK strip is X11-only behavior; force the X11 backend even
        // when a Wayland display is also reachable (KEYB_SHELL=x11).
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            use winit::platform::x11::EventLoopBuilderExtX11;
            EventLoop::builder().with_x11().build()?
        }
        #[cfg(any(not(unix), target_os = "macos"))]
        EventLoop::new()?
    } else {
        EventLoop::new()?
    };
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        core,
        gpu: Gpu::new(),
        state: None,
        error: None,
        osk,
    };
    event_loop.run_app(&mut app)?;
    match app.error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::clamp_osk_width;

    #[test]
    fn osk_width_preserves_distinct_toolbar_controls() {
        assert_eq!(clamp_osk_width(100, 1920, 1.0), 188);
        assert_eq!(clamp_osk_width(100, 3840, 2.0), 376);
        assert_eq!(clamp_osk_width(100, 80, 1.0), 80);
    }
}

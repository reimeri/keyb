//! keyb-ui: winit window + vello scene for the zooming canvas (SPEC §7).

mod font;
mod scene;

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use keyb_core::{Engine, PointerInput};
use vello::peniko::Color;
use vello::util::{RenderContext, RenderSurface};
use vello::{AaConfig, AaSupport, RenderParams, Renderer, RendererOptions, Scene};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Height of the committed-text ribbon (logical px).
pub const RIBBON_H: f64 = 56.0;
/// Origin line: pointer right of this fraction of the canvas zooms in,
/// left of it zooms out.
pub const ORIGIN_X_FRAC: f64 = 0.18;

struct RenderState {
    surface: RenderSurface<'static>,
    window: Arc<Window>,
}

struct App {
    context: RenderContext,
    renderers: Vec<Option<Renderer>>,
    state: Option<RenderState>,
    scene: Scene,
    engine: Engine,
    fonts: font::FontAssets,
    cursor: (f64, f64),
    engaged: bool,
    last_frame: Instant,
}

impl App {
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

    fn frame(&mut self) {
        let Some(state) = &self.state else { return };
        let size = state.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }
        let scale = state.window.scale_factor();
        let lw = size.width as f64 / scale;
        let lh = size.height as f64 / scale;

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f64().clamp(0.0, 0.05);
        self.last_frame = now;

        let input = self.pointer_input(lw, lh);
        self.engine.set_viewport_px(lh - RIBBON_H);
        self.engine.step(input, dt);

        self.scene.reset();
        scene::build(
            &mut self.scene,
            &self.engine,
            &self.fonts,
            lw,
            lh,
            scale,
            input,
        );

        let surface = &self.state.as_ref().unwrap().surface;
        let dev_id = surface.dev_id;
        let device_handle = &self.context.devices[dev_id];
        let surface_texture = match surface.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return, // outdated swapchain; next frame resizes
        };
        let renderer = self.renderers[dev_id].as_mut().unwrap();
        renderer
            .render_to_surface(
                &device_handle.device,
                &device_handle.queue,
                &self.scene,
                &surface_texture,
                &RenderParams {
                    base_color: Color::from_rgb8(0x14, 0x15, 0x20),
                    width: size.width,
                    height: size.height,
                    antialiasing_method: AaConfig::Area,
                },
            )
            .expect("vello render failed");
        surface_texture.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("keyb")
                        .with_inner_size(LogicalSize::new(1200.0, 800.0)),
                )
                .expect("create window"),
        );
        let size = window.inner_size();
        let surface = pollster::block_on(self.context.create_surface(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
            vello::wgpu::PresentMode::AutoVsync,
        ))
        .expect("create surface");
        self.renderers
            .resize_with(self.context.devices.len(), || None);
        let dev_id = surface.dev_id;
        self.renderers[dev_id].get_or_insert_with(|| {
            Renderer::new(
                &self.context.devices[dev_id].device,
                RendererOptions {
                    surface_format: Some(surface.format),
                    use_cpu: false,
                    antialiasing_support: AaSupport::area_only(),
                    num_init_threads: None,
                },
            )
            .expect("create renderer")
        });
        self.state = Some(RenderState { surface, window });
        self.last_frame = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    self.context
                        .resize_surface(&mut state.surface, size.width.max(1), size.height.max(1));
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(state) = &self.state {
                    let scale = state.window.scale_factor();
                    self.cursor = (position.x / scale, position.y / scale);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.engaged = state == ElementState::Pressed;
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => event_loop.exit(),
                        Key::Character(ref c) if c == "+" || c == "=" => {
                            self.engine.speed_bits = (self.engine.speed_bits + 0.5).min(10.0);
                        }
                        Key::Character(ref c) if c == "-" => {
                            self.engine.speed_bits = (self.engine.speed_bits - 0.5).max(1.0);
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.frame();
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

/// Run the UI until the window closes. Takes ownership of the engine.
pub fn run(engine: Engine) -> Result<()> {
    let fonts = font::FontAssets::load_system()?;
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        context: RenderContext::new(),
        renderers: Vec::new(),
        state: None,
        scene: Scene::new(),
        engine,
        fonts,
        cursor: (0.0, 0.0),
        engaged: false,
        last_frame: Instant::now(),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

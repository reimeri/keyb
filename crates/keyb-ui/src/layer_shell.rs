//! Wayland wlr-layer-shell frontend (SPEC §7.3): an overlay strip anchored
//! to the bottom edge with `keyboard_interactivity: none`, so the OSK can
//! *never* steal keyboard focus — clicks steer keyb while the compositor
//! keeps keyboard focus (and therefore text-input activation and synthesized
//! keys) on the target app.
//!
//! winit has no layer-shell support, so this frontend speaks smithay-client-
//! toolkit directly and hands vello a wgpu surface built from the raw
//! wl_surface handle. Redraw is driven by frame callbacks (vsync-paced).

use std::ptr::NonNull;

use anyhow::{anyhow, Context as _, Result};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_layer, delegate_output, delegate_pointer, delegate_registry,
    delegate_seat, delegate_shm,
    output::{OutputHandler, OutputState},
    registry::{ProvidesRegistryState, RegistryState},
    registry_handlers,
    seat::{
        pointer::{
            CursorIcon, PointerEvent, PointerEventKind, PointerHandler, ThemeSpec, ThemedPointer,
        },
        Capability, SeatHandler, SeatState,
    },
    shell::{
        wlr_layer::{
            Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
            LayerSurfaceConfigure,
        },
        WaylandSurface,
    },
    shm::{Shm, ShmHandler},
};
use vello::util::RenderSurface;
use wayland_client::{
    globals::registry_queue_init,
    protocol::{wl_output, wl_pointer, wl_seat, wl_surface},
    Connection, Proxy, QueueHandle,
};

use crate::{osk_height, Core, Gpu};

/// Left mouse button, evdev code (input-event-codes.h BTN_LEFT).
const BTN_LEFT: u32 = 0x110;

/// Raw-handle bridge from the layer surface to wgpu. Owning clones of the
/// connection and surface proxies keeps both alive as long as the wgpu
/// surface holds this.
struct WaylandHandles {
    conn: Connection,
    surface: wl_surface::WlSurface,
}

impl raw_window_handle::HasDisplayHandle for WaylandHandles {
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        let ptr = NonNull::new(self.conn.backend().display_ptr().cast())
            .ok_or(raw_window_handle::HandleError::Unavailable)?;
        let raw = raw_window_handle::WaylandDisplayHandle::new(ptr);
        Ok(unsafe {
            raw_window_handle::DisplayHandle::borrow_raw(raw_window_handle::RawDisplayHandle::Wayland(raw))
        })
    }
}

impl raw_window_handle::HasWindowHandle for WaylandHandles {
    fn window_handle(
        &self,
    ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        let ptr = NonNull::new(self.surface.id().as_ptr().cast())
            .ok_or(raw_window_handle::HandleError::Unavailable)?;
        let raw = raw_window_handle::WaylandWindowHandle::new(ptr);
        Ok(unsafe {
            raw_window_handle::WindowHandle::borrow_raw(raw_window_handle::RawWindowHandle::Wayland(raw))
        })
    }
}

struct AppState {
    core: Core,
    gpu: Gpu,
    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
    shm: Shm,
    compositor: CompositorState,
    layer: LayerSurface,
    pointer: Option<ThemedPointer>,
    render_surface: Option<RenderSurface<'static>>,
    /// Logical size from the last configure.
    width: u32,
    height: u32,
    scale: i32,
    configured: bool,
    exit: bool,
}

impl AppState {
    fn physical(&self) -> (u32, u32) {
        (
            (self.width * self.scale as u32).max(1),
            (self.height * self.scale as u32).max(1),
        )
    }

    fn resize(&mut self) {
        let (pw, ph) = self.physical();
        if let Some(rs) = &mut self.render_surface {
            self.gpu.context.resize_surface(rs, pw, ph);
        }
        self.layer.wl_surface().set_buffer_scale(self.scale);
    }

    fn draw(&mut self, qh: &QueueHandle<Self>) {
        if !self.configured || self.render_surface.is_none() {
            return;
        }
        self.core
            .tick(self.width as f64, self.height as f64, self.scale as f64);
        // Ask for the next vsync tick before presenting this frame.
        let s = self.layer.wl_surface();
        s.frame(qh, s.clone());
        let (pw, ph) = self.physical();
        let rs = self.render_surface.as_ref().expect("checked above");
        self.gpu.render(rs, &self.core.scene, pw, ph);
    }
}

impl CompositorHandler for AppState {
    fn scale_factor_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        new_factor: i32,
    ) {
        self.scale = new_factor.max(1);
        self.resize();
    }

    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: wl_output::Transform,
    ) {
    }

    fn frame(&mut self, _: &Connection, qh: &QueueHandle<Self>, _: &wl_surface::WlSurface, _: u32) {
        self.draw(qh);
    }

    fn surface_enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }

    fn surface_leave(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: &wl_output::WlOutput,
    ) {
    }
}

impl OutputHandler for AppState {
    fn output_state(&mut self) -> &mut OutputState {
        &mut self.output_state
    }

    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}

    fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}

    fn output_destroyed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
}

impl LayerShellHandler for AppState {
    fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &LayerSurface) {
        self.exit = true;
    }

    fn configure(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        _: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        let (w, h) = configure.new_size;
        self.width = if w == 0 { self.width.max(800) } else { w };
        self.height = if h == 0 { osk_height() } else { h };
        self.resize();
        let first = !self.configured;
        self.configured = true;
        if first {
            self.draw(qh);
        }
    }
}

impl SeatHandler for AppState {
    fn seat_state(&mut self) -> &mut SeatState {
        &mut self.seat_state
    }

    fn new_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}

    fn new_capability(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        seat: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Pointer && self.pointer.is_none() {
            let cursor_surface = self.compositor.create_surface(qh);
            match self.seat_state.get_pointer_with_theme(
                qh,
                &seat,
                self.shm.wl_shm(),
                cursor_surface,
                ThemeSpec::default(),
            ) {
                Ok(p) => self.pointer = Some(p),
                Err(e) => eprintln!("keyb: no pointer on seat: {e}"),
            }
        }
    }

    fn remove_capability(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: wl_seat::WlSeat,
        capability: Capability,
    ) {
        if capability == Capability::Pointer {
            self.pointer = None;
        }
    }

    fn remove_seat(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_seat::WlSeat) {}
}

impl PointerHandler for AppState {
    fn pointer_frame(
        &mut self,
        conn: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_pointer::WlPointer,
        events: &[PointerEvent],
    ) {
        for ev in events {
            if ev.surface != *self.layer.wl_surface() {
                continue;
            }
            match ev.kind {
                PointerEventKind::Enter { .. } => {
                    self.core.cursor = ev.position;
                    if let Some(p) = &self.pointer {
                        let _ = p.set_cursor(conn, CursorIcon::Default);
                    }
                }
                PointerEventKind::Leave { .. } => {
                    // No pointer capture yet: a drag that escapes the strip
                    // must not stay engaged forever.
                    self.core.engaged = false;
                }
                PointerEventKind::Motion { .. } => self.core.cursor = ev.position,
                PointerEventKind::Press { button, .. } if button == BTN_LEFT => {
                    self.core.engaged = true;
                }
                PointerEventKind::Release { button, .. } if button == BTN_LEFT => {
                    self.core.engaged = false;
                }
                PointerEventKind::Axis { vertical, .. } => {
                    let lines = if vertical.discrete != 0 {
                        vertical.discrete as f64
                    } else {
                        vertical.absolute / 15.0
                    };
                    // Scroll up = faster (wayland positive axis points down).
                    self.core.adjust_speed(-0.25 * lines);
                }
                _ => {}
            }
        }
    }
}

impl ShmHandler for AppState {
    fn shm_state(&mut self) -> &mut Shm {
        &mut self.shm
    }
}

impl ProvidesRegistryState for AppState {
    fn registry(&mut self) -> &mut RegistryState {
        &mut self.registry_state
    }

    registry_handlers![OutputState, SeatState];
}

delegate_compositor!(AppState);
delegate_output!(AppState);
delegate_shm!(AppState);
delegate_seat!(AppState);
delegate_pointer!(AppState);
delegate_layer!(AppState);
delegate_registry!(AppState);

/// On failure the core comes back so the caller can fall back to a window.
pub type LayerError = Box<(Core, anyhow::Error)>;

/// Run as a layer-shell overlay.
pub fn run(core: Core) -> Result<(), LayerError> {
    match setup(core) {
        Ok((mut state, mut queue)) => loop {
            if let Err(e) = queue.blocking_dispatch(&mut state) {
                return Err(Box::new((state.core, anyhow!(e).context("wayland dispatch"))));
            }
            if state.exit {
                return Ok(());
            }
        },
        Err(e) => Err(e),
    }
}

type Setup = (AppState, wayland_client::EventQueue<AppState>);

/// Everything fallible that doesn't need the core yet.
struct Parts {
    conn: Connection,
    queue: wayland_client::EventQueue<AppState>,
    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
    shm: Shm,
    compositor: CompositorState,
    layer: LayerSurface,
}

fn wayland_parts() -> Result<Parts> {
    let conn = Connection::connect_to_env().context("connect to wayland display")?;
    let (globals, queue) = registry_queue_init::<AppState>(&conn)?;
    let qh = queue.handle();
    let compositor = CompositorState::bind(&globals, &qh).context("wl_compositor")?;
    let layer_shell = LayerShell::bind(&globals, &qh).context("compositor lacks wlr-layer-shell")?;
    let shm = Shm::bind(&globals, &qh).context("wl_shm")?;

    let surface = compositor.create_surface(&qh);
    let layer =
        layer_shell.create_layer_surface(&qh, surface, Layer::Overlay, Some("keyb"), None);
    let h = osk_height();
    layer.set_anchor(Anchor::BOTTOM | Anchor::LEFT | Anchor::RIGHT);
    layer.set_size(0, h);
    // Reserve the strip so tiled/maximized targets shift up instead of
    // hiding their text field under the OSK.
    layer.set_exclusive_zone(h as i32);
    layer.set_keyboard_interactivity(KeyboardInteractivity::None);
    layer.commit();

    Ok(Parts {
        registry_state: RegistryState::new(&globals),
        seat_state: SeatState::new(&globals, &qh),
        output_state: OutputState::new(&globals, &qh),
        conn,
        queue,
        shm,
        compositor,
        layer,
    })
}

fn setup(core: Core) -> Result<Setup, LayerError> {
    let parts = match wayland_parts() {
        Ok(p) => p,
        Err(e) => return Err(Box::new((core, e))),
    };
    let mut gpu = Gpu::new();
    let handles = WaylandHandles {
        conn: parts.conn.clone(),
        surface: parts.layer.wl_surface().clone(),
    };
    let (w, h) = (800, osk_height());
    let render_surface = match pollster::block_on(gpu.context.create_surface(
        handles,
        w,
        h,
        vello::wgpu::PresentMode::AutoVsync,
    )) {
        Ok(s) => s,
        Err(e) => return Err(Box::new((core, anyhow!(e).context("create wgpu surface")))),
    };
    gpu.ensure_renderer(render_surface.dev_id, render_surface.format);
    Ok((
        AppState {
            core,
            gpu,
            registry_state: parts.registry_state,
            seat_state: parts.seat_state,
            output_state: parts.output_state,
            shm: parts.shm,
            compositor: parts.compositor,
            layer: parts.layer,
            pointer: None,
            render_surface: Some(render_surface),
            width: w,
            height: h,
            scale: 1,
            configured: false,
            exit: false,
        },
        parts.queue,
    ))
}

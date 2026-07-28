//! Wayland wlr-layer-shell frontend (SPEC §7.3): a full-height overlay column
//! anchored to the right edge with `keyboard_interactivity: none`, so the OSK can
//! *never* steal keyboard focus — clicks steer keyb while the compositor
//! keeps keyboard focus (and therefore text-input activation and synthesized
//! keys) on the target app.
//!
//! winit has no layer-shell support, so this frontend speaks smithay-client-
//! toolkit directly and hands vello a wgpu surface built from the raw
//! wl_surface handle. Redraw is driven by frame callbacks (vsync-paced).

use std::ptr::NonNull;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context as _, Result};
use smithay_client_toolkit::{
    compositor::{CompositorHandler, CompositorState},
    delegate_compositor, delegate_layer, delegate_output, delegate_pointer, delegate_registry,
    delegate_seat, delegate_shm,
    output::{OutputHandler, OutputState},
    reexports::{
        calloop::{channel, EventLoop},
        calloop_wayland_source::WaylandSource,
    },
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

use crate::scene::{hide_button_rect, reset_button_rect};
use crate::{osk_width, Core, Gpu, RenderOutcome, HANDLE_H, HANDLE_W};

/// Left mouse button, evdev code (input-event-codes.h BTN_LEFT).
const BTN_LEFT: u32 = 0x110;
const FOCUS_RETRY_MIN: Duration = Duration::from_secs(1);
const FOCUS_RETRY_MAX: Duration = Duration::from_secs(30);

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
            raw_window_handle::DisplayHandle::borrow_raw(
                raw_window_handle::RawDisplayHandle::Wayland(raw),
            )
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
            raw_window_handle::WindowHandle::borrow_raw(
                raw_window_handle::RawWindowHandle::Wayland(raw),
            )
        })
    }
}

struct FocusRetry {
    output: String,
    attempt_at: Instant,
    next_delay: Duration,
}

struct PendingLayer {
    // Drop the native surface before its layer-owned wl_surface.
    render_surface: RenderSurface<'static>,
    layer: LayerSurface,
    output: wl_output::WlOutput,
    output_height: u32,
    full_width: u32,
    scale: i32,
    name: String,
}

struct AppState {
    // Declaration order is drop order: wgpu must release its native surface
    // before the layer role destroys the underlying wl_surface.
    render_surface: Option<RenderSurface<'static>>,
    layer: LayerSurface,
    pending_layer: Option<PendingLayer>,
    core: Core,
    gpu: Gpu,
    registry_state: RegistryState,
    seat_state: SeatState,
    output_state: OutputState,
    shm: Shm,
    conn: Connection,
    qh: QueueHandle<Self>,
    compositor: CompositorState,
    layer_shell: LayerShell,
    pointer: Option<ThemedPointer>,
    /// Logical size from the last configure.
    width: u32,
    height: u32,
    /// Shown column width, preserved across a hide (while hidden, `width` holds
    /// the small handle size instead).
    full_width: u32,
    scale: i32,
    /// Output currently hosting the layer surface. It is learned from
    /// `surface_enter` for the compositor-selected initial placement.
    current_output: Option<wl_output::WlOutput>,
    /// Niri connector name most recently resolved from the focused app.
    desired_output: Option<String>,
    /// Newest queued focus change. Applying it after each calloop dispatch
    /// coalesces rapid focus changes into one surface migration.
    pending_output: Option<String>,
    /// Failed migrations retry with capped exponential backoff so a transient
    /// surface error cannot strand the keyboard on the old output.
    focus_retry: Option<FocusRetry>,
    configured: bool,
    exit: bool,
    render_error: Option<anyhow::Error>,
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

    fn output_named(&self, name: &str) -> Option<wl_output::WlOutput> {
        self.output_state.outputs().find(|output| {
            self.output_state
                .info(output)
                .and_then(|info| info.name)
                .as_deref()
                == Some(name)
        })
    }

    fn output_size(&self, output: &wl_output::WlOutput) -> Option<(u32, u32)> {
        let info = self.output_state.info(output)?;
        info.logical_size
            .filter(|(width, height)| *width > 0 && *height > 0)
            .map(|(width, height)| (width as u32, height as u32))
            .or_else(|| {
                let scale = info.scale_factor.max(1) as u32;
                info.modes
                    .iter()
                    .find(|mode| mode.current)
                    .and_then(|mode| {
                        let (width, height) = mode.dimensions;
                        (width > 0 && height > 0)
                            .then_some((width as u32 / scale, height as u32 / scale))
                    })
            })
    }

    fn output_width(&self, output: &wl_output::WlOutput) -> Option<u32> {
        self.output_size(output).map(|(width, _)| width)
    }

    /// Size the column to ~1/3 of its current output (unless `KEYB_WIDTH`
    /// pins it). Ignore updates for other monitors.
    fn apply_output_width(&mut self, output: &wl_output::WlOutput) {
        if self.current_output.as_ref() != Some(output) || std::env::var_os("KEYB_WIDTH").is_some()
        {
            return;
        }
        let Some(output_width) = self.output_width(output) else {
            return;
        };
        let want = osk_width(Some(output_width));
        self.full_width = want;
        // While hidden, only record the new width; it applies on the next show.
        if self.core.visible && want != self.width {
            self.width = want;
            self.layer.set_size(want, 0);
            self.layer.commit();
        }
    }

    /// Move the layer surface to Niri's focused application's output. Layer
    /// shell assigns an output only when a role is created, so migration means
    /// replacing the wl_surface and its wgpu surface while retaining `Core`.
    fn follow_focused_output(&mut self, name: String) -> Result<()> {
        self.desired_output = Some(name.clone());
        let Some(output) = self.output_named(&name) else {
            // Output metadata may still be arriving; OutputHandler retries.
            return Ok(());
        };
        if self.current_output.as_ref() == Some(&output)
            || self
                .pending_layer
                .as_ref()
                .is_some_and(|pending| pending.output == output)
        {
            return Ok(());
        }

        let info = self.output_state.info(&output);
        let full_width = osk_width(self.output_width(&output));
        let output_height = self
            .output_size(&output)
            .map(|(_, height)| height)
            .unwrap_or(self.height.max(1));
        let scale = info
            .as_ref()
            .map(|info| info.scale_factor.max(1))
            .unwrap_or(1);
        let (width, height) = if self.core.visible {
            (full_width, output_height)
        } else {
            (HANDLE_W, HANDLE_H)
        };

        let surface = self.compositor.create_surface(&self.qh);
        let layer = self.layer_shell.create_layer_surface(
            &self.qh,
            surface,
            Layer::Overlay,
            Some("keyb"),
            Some(&output),
        );
        if self.core.visible {
            layer.set_anchor(Anchor::RIGHT | Anchor::TOP | Anchor::BOTTOM);
            layer.set_size(full_width, 0);
        } else {
            layer.set_anchor(Anchor::RIGHT);
            layer.set_size(HANDLE_W, HANDLE_H);
        }
        layer.set_exclusive_zone(0);
        layer.set_keyboard_interactivity(KeyboardInteractivity::None);
        layer.wl_surface().set_buffer_scale(scale);
        layer.commit();

        let handles = WaylandHandles {
            conn: self.conn.clone(),
            surface: layer.wl_surface().clone(),
        };
        let render_surface = pollster::block_on(self.gpu.context.create_surface(
            handles,
            (width * scale as u32).max(1),
            (height * scale as u32).max(1),
            vello::wgpu::PresentMode::AutoVsync,
        ))
        .context("create render surface on focused output")?;
        self.gpu
            .ensure_renderer(render_surface.dev_id, render_surface.format)
            .context("initialize renderer on focused output")?;

        // Keep the current, valid layer mapped until the compositor accepts
        // and configures the replacement. Replacing an older pending move is
        // safe because PendingLayer drops its native surface before its role.
        self.pending_layer = Some(PendingLayer {
            render_surface,
            layer,
            output,
            output_height,
            full_width,
            scale,
            name,
        });
        Ok(())
    }

    fn schedule_focus_retry(&mut self, output: String, delay: Duration) {
        self.focus_retry = Some(FocusRetry {
            output,
            attempt_at: Instant::now() + delay,
            next_delay: (delay * 2).min(FOCUS_RETRY_MAX),
        });
    }

    fn finish_pending_layer(&mut self, configure: LayerSurfaceConfigure, qh: &QueueHandle<Self>) {
        let Some(pending) = self.pending_layer.take() else {
            return;
        };
        let PendingLayer {
            render_surface,
            layer,
            output,
            output_height,
            full_width,
            scale,
            name,
        } = pending;
        let fallback = if self.core.visible {
            (full_width, output_height)
        } else {
            (HANDLE_W, HANDLE_H)
        };
        let (width, height) = configure.new_size;

        // The replacement has now been accepted by the compositor. Retire the
        // old native surface before destroying its layer-owned wl_surface.
        let old_render_surface = self.render_surface.replace(render_surface);
        let old_layer = std::mem::replace(&mut self.layer, layer);
        self.core.engaged = false;
        self.width = if width == 0 { fallback.0 } else { width };
        self.height = if height == 0 { fallback.1 } else { height };
        self.full_width = full_width;
        self.scale = scale;
        self.current_output = Some(output);
        self.configured = true;
        self.focus_retry = None;
        self.resize();
        drop(old_render_surface);
        drop(old_layer);
        eprintln!("keyb: moved keyboard to focused output {name}");
        self.draw(qh);
    }

    fn reconsider_output(&mut self, output: &wl_output::WlOutput) {
        let output_name = self.output_state.info(output).and_then(|info| info.name);
        if output_name.as_deref() == self.desired_output.as_deref()
            && self.current_output.as_ref() != Some(output)
        {
            if let Some(name) = output_name {
                if let Err(error) = self.follow_focused_output(name.clone()) {
                    eprintln!("keyb: could not move keyboard to focused output: {error:#}");
                    self.schedule_focus_retry(name, FOCUS_RETRY_MIN);
                }
            }
        } else {
            self.apply_output_width(output);
        }
    }

    /// Toggle the OSK between the full column and the collapsed edge handle by
    /// resizing the layer surface. The following `configure` carries the new
    /// size and repaints.
    fn set_visible(&mut self, vis: bool) {
        if self.core.visible == vis {
            return;
        }
        self.core.visible = vis;
        self.core.engaged = false;
        if vis {
            self.layer
                .set_anchor(Anchor::RIGHT | Anchor::TOP | Anchor::BOTTOM);
            self.layer.set_size(self.full_width, 0);
        } else {
            self.layer.set_anchor(Anchor::RIGHT);
            self.layer.set_size(HANDLE_W, HANDLE_H);
        }
        self.layer.commit();
    }

    fn draw(&mut self, qh: &QueueHandle<Self>) {
        if !self.configured || self.render_surface.is_none() {
            return;
        }
        self.core
            .tick(self.width as f64, self.height as f64, self.scale as f64);
        // Ask for the next vsync tick only while visible; the collapsed handle
        // is static, so hidden draws once and the loop goes idle.
        let frame_requested = self.core.visible;
        if frame_requested {
            let s = self.layer.wl_surface();
            s.frame(qh, s.clone());
        }
        let (pw, ph) = self.physical();
        let Some(render_surface) = self.render_surface.as_mut() else {
            return;
        };
        let result = self.gpu.render(render_surface, &self.core.scene, pw, ph);
        match result {
            Ok(RenderOutcome::Presented) => {}
            Ok(RenderOutcome::Retry) => {
                // Frame callbacks take effect on a surface commit. Acquisition
                // failed before wgpu could commit a buffer, so commit the
                // callback explicitly to guarantee another draw attempt.
                let surface = self.layer.wl_surface();
                if !frame_requested {
                    surface.frame(qh, surface.clone());
                }
                self.layer.commit();
            }
            Err(error) => {
                self.core.engaged = false;
                self.render_error = Some(error.context("render layer-shell surface"));
                self.exit = true;
            }
        }
    }
}

impl CompositorHandler for AppState {
    fn scale_factor_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        new_factor: i32,
    ) {
        if surface == self.layer.wl_surface() {
            self.scale = new_factor.max(1);
            self.resize();
        }
    }

    fn transform_changed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        _: &wl_surface::WlSurface,
        _: wl_output::Transform,
    ) {
    }

    fn frame(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        _: u32,
    ) {
        if surface == self.layer.wl_surface() {
            self.draw(qh);
        }
    }

    fn surface_enter(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        surface: &wl_surface::WlSurface,
        output: &wl_output::WlOutput,
    ) {
        if surface != self.layer.wl_surface() {
            return;
        }
        if self.current_output.is_none() {
            self.current_output = Some(output.clone());
        }
        self.apply_output_width(output);
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

    fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, output: wl_output::WlOutput) {
        self.reconsider_output(&output);
    }

    fn update_output(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        self.reconsider_output(&output);
    }

    fn output_destroyed(
        &mut self,
        _: &Connection,
        _: &QueueHandle<Self>,
        output: wl_output::WlOutput,
    ) {
        if self.current_output.as_ref() == Some(&output) {
            self.current_output = None;
        }
    }
}

impl LayerShellHandler for AppState {
    fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, layer: &LayerSurface) {
        if self
            .pending_layer
            .as_ref()
            .is_some_and(|pending| &pending.layer == layer)
        {
            let pending = self
                .pending_layer
                .take()
                .expect("pending layer disappeared");
            eprintln!(
                "keyb: compositor rejected move to focused output {}; retrying",
                pending.name
            );
            self.schedule_focus_retry(pending.name.clone(), FOCUS_RETRY_MIN);
            return;
        }
        if layer == &self.layer {
            self.exit = true;
        }
    }

    fn configure(
        &mut self,
        _: &Connection,
        qh: &QueueHandle<Self>,
        layer: &LayerSurface,
        configure: LayerSurfaceConfigure,
        _serial: u32,
    ) {
        if self
            .pending_layer
            .as_ref()
            .is_some_and(|pending| &pending.layer == layer)
        {
            self.finish_pending_layer(configure, qh);
            return;
        }
        if layer != &self.layer {
            return;
        }
        let (w, h) = configure.new_size;
        // Column: width from us, full height from the compositor. Keep the
        // last-known value on a 0 (which shouldn't happen under TOP|BOTTOM).
        self.width = if w == 0 { self.width } else { w };
        self.height = if h == 0 { self.height } else { h };
        self.resize();
        self.configured = true;
        // Repaint on every configure so a visibility toggle's resize is drawn
        // (hidden draws don't re-arm frames, so this is the only redraw then).
        self.draw(qh);
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
                    let (cx, cy) = self.core.cursor;
                    if !self.core.visible {
                        // Any tap on the collapsed handle reopens the OSK.
                        self.set_visible(true);
                    } else if hide_button_rect(self.width as f64)
                        .contains(vello::kurbo::Point::new(cx, cy))
                    {
                        self.set_visible(false);
                    } else if reset_button_rect(self.width as f64)
                        .contains(vello::kurbo::Point::new(cx, cy))
                    {
                        self.core.reset();
                    } else {
                        self.core.engaged = true;
                    }
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

/// Run as a layer-shell overlay. Redraw is frame-callback driven while the OSK
/// is shown. A calloop source also wakes the loop for Niri focus changes, even
/// while the collapsed handle is otherwise idle.
pub fn run(core: Core) -> Result<(), LayerError> {
    let (mut state, queue) = setup(core)?;
    let mut event_loop = match EventLoop::<AppState>::try_new() {
        Ok(event_loop) => event_loop,
        Err(error) => {
            return Err(Box::new((
                state.core,
                anyhow!(error).context("create layer-shell event loop"),
            )))
        }
    };
    // Do one dispatch pass per readiness notification. WaylandSource's
    // convenience `insert` drains until no event remains; continuous frame
    // callbacks can then starve the Niri channel on fast compositors.
    if let Err(error) = event_loop.handle().insert_source(
        WaylandSource::new(state.conn.clone(), queue),
        |_, queue, state: &mut AppState| queue.dispatch_pending(state).map(|_| 0),
    ) {
        return Err(Box::new((
            state.core,
            anyhow!(error.error).context("register Wayland event source"),
        )));
    }
    if let Some(focus_events) = crate::niri_focus::subscribe() {
        if let Err(error) =
            event_loop
                .handle()
                .insert_source(focus_events, |event, _, state: &mut AppState| {
                    if let channel::Event::Msg(output) = event {
                        state.pending_output = Some(output);
                        state.focus_retry = None;
                    }
                })
        {
            return Err(Box::new((
                state.core,
                anyhow!(error.error).context("register Niri focus event source"),
            )));
        }
    }

    loop {
        let timeout = state
            .focus_retry
            .as_ref()
            .map(|retry| retry.attempt_at.saturating_duration_since(Instant::now()));
        if let Err(error) = event_loop.dispatch(timeout, &mut state) {
            return Err(Box::new((
                state.core,
                anyhow!(error).context("layer-shell event dispatch"),
            )));
        }
        if let Some(error) = state.render_error.take() {
            return Err(Box::new((state.core, error)));
        }
        let migration = state
            .pending_output
            .take()
            .map(|output| (output, FOCUS_RETRY_MIN))
            .or_else(|| {
                let retry = state.focus_retry.take()?;
                if retry.attempt_at > Instant::now() {
                    state.focus_retry = Some(retry);
                    return None;
                }
                Some((retry.output, retry.next_delay))
            });
        if let Some((output, retry_delay)) = migration {
            let retry_output = output.clone();
            if let Err(error) = state.follow_focused_output(output) {
                // The old layer and render surfaces remain valid until a
                // replacement is fully initialized. Keep the non-focusable
                // OSK and retry with bounded backoff.
                eprintln!("keyb: could not move keyboard to focused output: {error:#}");
                state.focus_retry = Some(FocusRetry {
                    output: retry_output,
                    attempt_at: Instant::now() + retry_delay,
                    next_delay: (retry_delay * 2).min(FOCUS_RETRY_MAX),
                });
            }
        }
        if state.exit {
            return Ok(());
        }
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
    layer_shell: LayerShell,
    layer: LayerSurface,
}

fn wayland_parts() -> Result<Parts> {
    let conn = Connection::connect_to_env().context("connect to wayland display")?;
    let (globals, queue) = registry_queue_init::<AppState>(&conn)?;
    let qh = queue.handle();
    let compositor = CompositorState::bind(&globals, &qh).context("wl_compositor")?;
    let layer_shell =
        LayerShell::bind(&globals, &qh).context("compositor lacks wlr-layer-shell")?;
    let shm = Shm::bind(&globals, &qh).context("wl_shm")?;

    let surface = compositor.create_surface(&qh);
    let layer = layer_shell.create_layer_surface(&qh, surface, Layer::Overlay, Some("keyb"), None);
    let w = osk_width(None);
    // Full-height column on the right edge. Width fixed; height 0 = let the
    // compositor fill top-to-bottom.
    layer.set_anchor(Anchor::RIGHT | Anchor::TOP | Anchor::BOTTOM);
    layer.set_size(w, 0);
    // Overlay: reserve nothing so tiled/maximized windows keep their full size
    // and the column floats over them.
    layer.set_exclusive_zone(0);
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
        layer_shell,
        layer,
    })
}

fn setup(core: Core) -> Result<Setup, LayerError> {
    let parts = match wayland_parts() {
        Ok(p) => p,
        Err(e) => return Err(Box::new((core, e))),
    };
    let qh = parts.queue.handle();
    let mut gpu = Gpu::new();
    let handles = WaylandHandles {
        conn: parts.conn.clone(),
        surface: parts.layer.wl_surface().clone(),
    };
    // Placeholder until the first configure; corrected once the output size
    // is known (`apply_output_width`) and the compositor sends the real height.
    let (w, h) = (osk_width(None), 1080);
    let render_surface = match pollster::block_on(gpu.context.create_surface(
        handles,
        w,
        h,
        vello::wgpu::PresentMode::AutoVsync,
    )) {
        Ok(s) => s,
        Err(e) => return Err(Box::new((core, anyhow!(e).context("create wgpu surface")))),
    };
    if let Err(error) = gpu.ensure_renderer(render_surface.dev_id, render_surface.format) {
        return Err(Box::new((
            core,
            error.context("initialize layer-shell renderer"),
        )));
    }
    Ok((
        AppState {
            core,
            gpu,
            registry_state: parts.registry_state,
            seat_state: parts.seat_state,
            output_state: parts.output_state,
            shm: parts.shm,
            conn: parts.conn,
            qh,
            compositor: parts.compositor,
            layer_shell: parts.layer_shell,
            layer: parts.layer,
            pending_layer: None,
            pointer: None,
            render_surface: Some(render_surface),
            width: w,
            height: h,
            full_width: w,
            scale: 1,
            current_output: None,
            desired_output: None,
            pending_output: None,
            focus_retry: None,
            configured: false,
            exit: false,
            render_error: None,
        },
        parts.queue,
    ))
}

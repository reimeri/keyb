//! Best-effort focused-output tracking for the Niri compositor.
//!
//! Niri exposes focus state over a newline-delimited JSON IPC stream. Keep the
//! IPC reader off the render thread and wake the calloop event loop only when
//! the focused application's output changes.

use std::collections::HashMap;
use std::ffi::OsString;
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::thread;
use std::time::Duration;

use serde_json::Value;
use smithay_client_toolkit::reexports::calloop::channel::{self, Channel, Sender};

const NIRI_SOCKET: &str = "NIRI_SOCKET";
const RETRY_DELAY: Duration = Duration::from_secs(1);

/// Start following Niri when this process was launched inside a Niri session.
/// The returned calloop source emits output connector names (for example
/// `DP-1`). Unknown future IPC events are deliberately ignored.
pub fn subscribe() -> Option<Channel<String>> {
    let socket_path = std::env::var_os(NIRI_SOCKET)?;
    let (sender, channel) = channel::channel();
    if let Err(error) = thread::Builder::new()
        .name("keyb-niri-focus".into())
        .spawn(move || follow_forever(socket_path, sender))
    {
        eprintln!("keyb: cannot start Niri focus worker: {error}");
        return None;
    }
    Some(channel)
}

fn follow_forever(socket_path: OsString, sender: Sender<String>) {
    let mut reported_error = false;
    loop {
        match follow_once(Path::new(&socket_path), &sender) {
            Ok(()) => return, // receiver was dropped; keyb is exiting
            Err(error) => {
                if !reported_error {
                    eprintln!(
                        "keyb: cannot follow Niri focused output ({error}); retrying in the background"
                    );
                    reported_error = true;
                }
            }
        }
        thread::sleep(RETRY_DELAY);
    }
}

/// Read one connection until it fails. `Ok` means the UI receiver went away.
fn follow_once(path: &Path, sender: &Sender<String>) -> io::Result<()> {
    let mut stream = BufReader::new(UnixStream::connect(path)?);
    stream.get_mut().write_all(b"\"EventStream\"\n")?;

    let mut line = String::new();
    read_line(&mut stream, &mut line)?;
    let reply: Value = serde_json::from_str(&line).map_err(io::Error::other)?;
    if reply.pointer("/Ok").and_then(Value::as_str) != Some("Handled") {
        return Err(io::Error::other(format!(
            "Niri rejected EventStream: {reply}"
        )));
    }

    let mut state = FocusState::default();
    loop {
        read_line(&mut stream, &mut line)?;
        let event: Value = match serde_json::from_str(&line) {
            Ok(event) => event,
            Err(error) => {
                // A malformed or newer event must not stop tracking all
                // subsequent focus changes.
                eprintln!("keyb: ignoring unrecognized Niri IPC event: {error}");
                continue;
            }
        };
        if let Some(output) = state.apply(&event) {
            if sender.send(output).is_err() {
                return Ok(());
            }
        }
    }
}

fn read_line(reader: &mut impl BufRead, line: &mut String) -> io::Result<()> {
    line.clear();
    if reader.read_line(line)? == 0 {
        Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "Niri IPC stream closed",
        ))
    } else {
        Ok(())
    }
}

#[derive(Debug, Default)]
struct FocusState {
    workspaces: HashMap<u64, Option<String>>,
    windows: HashMap<u64, Option<u64>>,
    focused_window: Option<u64>,
    last_output: Option<String>,
}

impl FocusState {
    /// Apply one Niri event and return a newly focused output, if it changed.
    fn apply(&mut self, event: &Value) -> Option<String> {
        if let Some(payload) = event.get("WorkspacesChanged") {
            self.workspaces.clear();
            if let Some(workspaces) = payload.get("workspaces").and_then(Value::as_array) {
                for workspace in workspaces {
                    if let Some(id) = workspace.get("id").and_then(Value::as_u64) {
                        let output = workspace
                            .get("output")
                            .and_then(Value::as_str)
                            .map(str::to_owned);
                        self.workspaces.insert(id, output);
                    }
                }
            }
        } else if let Some(payload) = event.get("WindowsChanged") {
            self.windows.clear();
            self.focused_window = None;
            if let Some(windows) = payload.get("windows").and_then(Value::as_array) {
                for window in windows {
                    let Some(id) = window.get("id").and_then(Value::as_u64) else {
                        continue;
                    };
                    let workspace_id = window.get("workspace_id").and_then(Value::as_u64);
                    self.windows.insert(id, workspace_id);
                    if window
                        .get("is_focused")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                    {
                        self.focused_window = Some(id);
                    }
                }
            }
        } else if let Some(payload) = event.get("WindowOpenedOrChanged") {
            if let Some(window) = payload.get("window") {
                if let Some(id) = window.get("id").and_then(Value::as_u64) {
                    let workspace_id = window.get("workspace_id").and_then(Value::as_u64);
                    self.windows.insert(id, workspace_id);
                    if window
                        .get("is_focused")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                    {
                        self.focused_window = Some(id);
                    }
                }
            }
        } else if let Some(payload) = event.get("WindowClosed") {
            if let Some(id) = payload.get("id").and_then(Value::as_u64) {
                self.windows.remove(&id);
                if self.focused_window == Some(id) {
                    self.focused_window = None;
                }
            }
        } else if let Some(payload) = event.get("WindowFocusChanged") {
            self.focused_window = payload.get("id").and_then(Value::as_u64);
        }

        let output = self
            .focused_window
            .and_then(|id| self.windows.get(&id).copied().flatten())
            .and_then(|workspace_id| self.workspaces.get(&workspace_id).cloned().flatten());
        if output != self.last_output {
            self.last_output = output.clone();
            output
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FocusState;
    use serde_json::json;

    fn initial_state(state: &mut FocusState) -> Option<String> {
        assert_eq!(
            state.apply(&json!({
                "WorkspacesChanged": {"workspaces": [
                    {"id": 10, "output": "DP-1"},
                    {"id": 20, "output": "HDMI-A-1"}
                ]}
            })),
            None
        );
        state.apply(&json!({
            "WindowsChanged": {"windows": [
                {"id": 100, "workspace_id": 10, "is_focused": true},
                {"id": 200, "workspace_id": 20, "is_focused": false}
            ]}
        }))
    }

    #[test]
    fn resolves_initial_focused_window_to_its_output() {
        let mut state = FocusState::default();
        assert_eq!(initial_state(&mut state).as_deref(), Some("DP-1"));
    }

    #[test]
    fn emits_only_when_focus_moves_to_another_output() {
        let mut state = FocusState::default();
        initial_state(&mut state);

        assert_eq!(
            state
                .apply(&json!({"WindowFocusChanged": {"id": 200}}))
                .as_deref(),
            Some("HDMI-A-1")
        );
        assert_eq!(
            state.apply(&json!({"WindowFocusChanged": {"id": 200}})),
            None
        );
    }

    #[test]
    fn follows_a_focused_window_moved_between_workspaces() {
        let mut state = FocusState::default();
        initial_state(&mut state);

        assert_eq!(
            state
                .apply(&json!({
                    "WindowOpenedOrChanged": {"window": {
                        "id": 100, "workspace_id": 20, "is_focused": true
                    }}
                }))
                .as_deref(),
            Some("HDMI-A-1")
        );
    }

    #[test]
    fn follows_a_workspace_reassigned_to_another_output() {
        let mut state = FocusState::default();
        initial_state(&mut state);

        assert_eq!(
            state
                .apply(&json!({
                    "WorkspacesChanged": {"workspaces": [
                        {"id": 10, "output": "HDMI-A-1"},
                        {"id": 20, "output": "DP-1"}
                    ]}
                }))
                .as_deref(),
            Some("HDMI-A-1")
        );
    }

    #[test]
    fn replacement_window_snapshot_discards_stale_mappings() {
        let mut state = FocusState::default();
        initial_state(&mut state);

        assert_eq!(
            state
                .apply(&json!({
                    "WindowsChanged": {"windows": [
                        {"id": 300, "workspace_id": 20, "is_focused": true}
                    ]}
                }))
                .as_deref(),
            Some("HDMI-A-1")
        );
        assert!(!state.windows.contains_key(&100));
    }

    #[test]
    fn tolerates_unknown_events_and_missing_focus() {
        let mut state = FocusState::default();
        initial_state(&mut state);

        assert_eq!(state.apply(&json!({"FutureEvent": {"field": 1}})), None);
        assert_eq!(
            state.apply(&json!({"WindowFocusChanged": {"id": null}})),
            None
        );
    }
}

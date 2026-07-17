//! Integration test (SPEC §11): round-trip commit + retract through a real
//! compositor into a real terminal. Requires `sway` and `foot`; skips with a
//! note when either is missing (CI images must install them).
//!
//! sway runs headless with an empty config; foot runs `cat > file`; the
//! injector connects as a second client. Asserts the file contents — i.e.
//! the full path compositor → terminal → pty, not just protocol acceptance.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use keyb_inject::{Injector, WaylandInjector};

fn have(bin: &str) -> bool {
    Command::new(bin)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

struct Sway {
    child: Child,
    display: String,
    ipc: PathBuf,
}

impl Drop for Sway {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start_sway() -> Sway {
    let runtime = std::env::var("XDG_RUNTIME_DIR").expect("XDG_RUNTIME_DIR");
    let child = Command::new("sway")
        .args(["-c", "/dev/null"])
        .env("WLR_BACKENDS", "headless")
        .env("WLR_LIBINPUT_NO_DEVICES", "1")
        .env_remove("WAYLAND_DISPLAY")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn sway");
    let uid = unsafe { libc::getuid() };
    let ipc = PathBuf::from(&runtime).join(format!("sway-ipc.{uid}.{}.sock", child.id()));
    // sway doesn't announce its display name at default log level, and
    // display sockets get reused across instances, so the only reliable
    // source is sway itself: exec a child that dumps $WAYLAND_DISPLAY.
    let probe = PathBuf::from(&runtime).join(format!("keyb-e2e-display-{}", child.id()));
    let _ = std::fs::remove_file(&probe);
    let mut sway = Sway {
        child,
        display: String::new(),
        ipc,
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    while !sway.ipc.exists() {
        assert!(Instant::now() < deadline, "sway IPC socket never appeared");
        std::thread::sleep(Duration::from_millis(100));
    }
    swaymsg(
        &sway,
        &[
            "exec",
            &format!("sh -c 'echo $WAYLAND_DISPLAY > {}'", probe.display()),
        ],
    );
    sway.display = loop {
        let s = std::fs::read_to_string(&probe).unwrap_or_default();
        if !s.trim().is_empty() {
            break s.trim().to_string();
        }
        assert!(Instant::now() < deadline, "sway never reported its display");
        std::thread::sleep(Duration::from_millis(100));
    };
    let _ = std::fs::remove_file(&probe);
    sway
}

fn swaymsg(sway: &Sway, args: &[&str]) -> String {
    let out = Command::new("swaymsg")
        .arg("-s")
        .arg(&sway.ipc)
        .args(args)
        .output()
        .expect("swaymsg");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Launch foot writing its stdin to `out` and wait until it has focus.
fn start_foot(sway: &Sway, out: &std::path::Path) {
    swaymsg(
        sway,
        &["exec", &format!("foot sh -c 'cat > {}'", out.display())],
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if swaymsg(sway, &["-t", "get_tree"]).contains("\"app_id\": \"foot\"") {
            // One settle beat: foot must also take keyboard focus and enable
            // its text input before injection starts.
            std::thread::sleep(Duration::from_millis(300));
            return;
        }
        assert!(Instant::now() < deadline, "foot never appeared");
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn read_out(path: &std::path::Path) -> String {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let s = std::fs::read_to_string(path).unwrap_or_default();
        if !s.is_empty() {
            return s;
        }
        assert!(
            Instant::now() < deadline,
            "no text arrived in {}",
            path.display()
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

const CHARS: &[char] = &[
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '.', ',', '\'', '\n',
];

fn run_case(name: &str, make: impl Fn() -> WaylandInjector) -> String {
    let sway = start_sway();
    let out = std::env::temp_dir().join(format!("keyb-e2e-{name}-{}.txt", std::process::id()));
    let _ = std::fs::remove_file(&out);
    start_foot(&sway, &out);
    // The injector reads WAYLAND_DISPLAY; tests in this file run serially in
    // one process, and nothing else here touches the variable.
    std::env::set_var("WAYLAND_DISPLAY", &sway.display);
    let mut inj = make();
    inj.commit("the quick fox").expect("commit");
    inj.retract("fox").expect("retract");
    inj.commit("cat\n").expect("commit 2");
    let text = read_out(&out);
    let _ = std::fs::remove_file(&out);
    text
}

#[test]
fn commit_and_retract_reach_a_real_terminal() {
    if !have("sway") || !have("foot") {
        eprintln!("SKIP: sway/foot not installed");
        return;
    }
    // Virtual-keyboard path (key synthesis + BackSpace retraction).
    let vk = run_case("vk", || {
        WaylandInjector::vk_only(CHARS).expect("vk injector")
    });
    assert_eq!(vk, "the quick cat\n", "virtual-keyboard path");
    // Hybrid path: commits ride input-method-v2 (foot speaks text-input-v3);
    // retraction must detect that a terminal has no surrounding text and
    // fall back to synthesized BackSpace.
    let im = run_case("im", || WaylandInjector::new(CHARS).expect("im injector"));
    assert_eq!(im, "the quick cat\n", "input-method hybrid path");
}

//! Injection driver for integration tests: applies `commit:<text>` /
//! `retract:<text>` args in order through the auto-detected backend
//! (`KEYB_INJECT` forces one). `\n` in args stands for newline.
//!
//! CAUTION: this types into whatever has keyboard focus on $WAYLAND_DISPLAY /
//! $DISPLAY. Meant for nested/headless compositors.

fn main() -> anyhow::Result<()> {
    let chars: Vec<char> = ('a'..='z').chain([' ', '.', ',', '\'', '\n']).collect();
    let mut inj = keyb_inject::detect(&chars);
    eprintln!("type_text: backend {:?}", inj.capabilities());
    for arg in std::env::args().skip(1) {
        let arg = arg.replace("\\n", "\n");
        if let Some(t) = arg.strip_prefix("commit:") {
            inj.commit(t)?;
        } else if let Some(t) = arg.strip_prefix("retract:") {
            inj.retract(t)?;
        } else {
            anyhow::bail!("args must be commit:<text> or retract:<text>");
        }
        // Give the compositor time to forward events between operations.
        std::thread::sleep(std::time::Duration::from_millis(80));
    }
    Ok(())
}

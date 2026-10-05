# keyb

A mouse-driven, zooming on-screen keyboard in the lineage of
[Dasher](https://en.wikipedia.org/wiki/Dasher_(software)), with a local LLM as the
probability model. You write by steering through nested boxes whose sizes are the
model's predictive distribution — no clicking through keys. Committed text is
injected into whatever application has focus (Wayland or X11).

[SPEC.md](SPEC.md) is the design document and the source of truth for behaviour.

## Requirements

- Linux with a graphical Wayland or X11 session; headless use is not supported.
- Stable Rust, a C/C++ toolchain, CMake, and libclang. `llama-cpp-2` builds
  llama.cpp and generates bindings locally.
- A working wgpu-compatible GPU/driver and a sans-serif system font. Fontconfig's
  `fc-match` is preferred; common DejaVu and Liberation Sans paths are fallbacks.
- For the default LLM, an internet connection and about 491 MB of disk space on
  first run. The app remains usable with its built-in PPM model without either.

Environment-specific support:

- **Wayland overlay:** requires the compositor to expose `wlr-layer-shell`;
  otherwise keyb falls back to a regular window. The Copy button additionally
  needs `ext-data-control` or `wlr-data-control`; when neither is available it
  falls back to X11 clipboard access if `DISPLAY` is set, or reports an error to
  stderr. Injection prefers
  `input-method-v2` or `virtual-keyboard-v1`; if neither is available,
  auto-detection tries X11/XTEST when `DISPLAY` is set, then the non-injecting
  buffer. XTEST reaches only X11/Xwayland clients.
- **X11:** the dock-style window and text injection require X11, with the XTEST
  extension needed for injection. Characters must exist in the active keyboard
  layout (unshifted or shifted).
- **Regular window:** works on Wayland or X11 and is useful on compositors
  without layer-shell, but it does not remove the injection requirements above.
- **Niri:** focused-monitor following is available only in the layer-shell
  frontend when `NIRI_SOCKET` is present.

## NixOS

Flake outputs: `packages.default` (`keyb`) and `devShells.default`. Linux only (`x86_64-linux`, `aarch64-linux`).

```bash
nix develop          # toolchain + native libs; then ./start.sh
nix run              # run the packaged binary
nix build            # ./result/bin/keyb
```

Add this repo as an input to your NixOS flake and make keyb follow your own
`nixpkgs` input:

```nix
inputs.keyb = {
  url = "github:reimeri/keyb";
  inputs.nixpkgs.follows = "nixpkgs";
};
```

This uses the same nixpkgs revision as your system, keeping keyb's runtime
libraries compatible with the system graphics drivers. Using keyb's separate
nixpkgs pin can cause glibc version mismatches that prevent EGL or Vulkan drivers
from loading, resulting in startup failures such as "Couldn't find suitable
device".

Then install the package:

```nix
environment.systemPackages = [ inputs.keyb.packages.${pkgs.system}.default ];
```

The package wraps `fc-match` and a DejaVu font so `nix run` can find a sans-serif face. A graphical Wayland or X11 session with working GPU drivers (`hardware.graphics.enable`) is still required; the flake does not configure the compositor.

On first run the default GGUF is still downloaded to `$XDG_DATA_HOME/keyb/models/` unless `KEYB_NO_MODEL_DOWNLOAD` is set or `KEYB_MODEL` points at an existing file.

## Run

```bash
./start.sh
```

That is `cargo run -p keyb-app --bin keyb`. On first run it downloads the default
model (Qwen2.5-0.5B-Instruct Q4_K_M, ~491 MB) to
`$XDG_DATA_HOME/keyb/models/` (or `$HOME/.local/share/keyb/models/` when unset).
If the download or load fails, keyb falls back to the built-in PPM character
model and still works.

Steering:

- In the layer-shell frontend on **Niri**, the keyboard follows the
  keyboard-focused application to its monitor when `NIRI_SOCKET` is available.
  Moving focus to an app on another output moves the keyboard there without
  resetting the current text or zoom
  state. If Niri IPC is temporarily unavailable, the keyboard stays on its
  last output while the connection retries in the background.
- **Hold the left button** to engage; release to stop.
- **Right of centre = write, left = back.** Backward zoom is the only undo, and
  zooming back across a commit boundary retracts already-injected text.
- **Vertical position** selects which box you drift toward.
- Scroll wheel or `+`/`-` adjusts speed; `Esc` quits. In the ribbon, **↺** resets
  the current context, **›** collapses to the edge handle, **⎘** copies all
  finalized text to the clipboard, **⧉** captures the primary selection
  (clipboard as fallback) as prediction context, and **×** quits keyb
  completely. While a context block is active a `ctx N` chip appears: left-click
  pins or dismisses the preview, right-click drops the block without resetting
  typed text.

Typed words are learned into a decaying personal vocabulary at
`$XDG_DATA_HOME/keyb/user_model.v1.bin` (or
`$HOME/.local/share/keyb/user_model.v1.bin` when unset). The last captured
context block is also dumped to `$XDG_STATE_HOME/keyb/last_context.txt` (or
`$HOME/.local/state/keyb/last_context.txt`) so you can check exactly what the
model was promised.

## Environment variables

| Variable | Effect |
|---|---|
| `KEYB_MODEL` | Path to a GGUF model instead of the managed default (disables auto-download). |
| `KEYB_NO_MODEL_DOWNLOAD` | Skip the default-model download; use PPM if the file is absent. |
| `KEYB_NO_PERSONALIZE` | Disable user-vocabulary learning. |
| `KEYB_ADAPTIVE` | Enable Dasher-style adaptive speed control (off by default). |
| `KEYB_SHELL` | `layer` \| `x11` \| `window` — window backend; auto-detected otherwise. `window` gives a plain resizable development window. |
| `KEYB_INJECT` | `wayland` \| `wayland-vk` \| `x11` \| `buffer` — text output backend. `buffer` injects nothing and is the fallback when auto-detection fails. |
| `KEYB_WIDTH` | Strip width in logical px (default: one third of the monitor). |

## Development

Cargo workspace; `keyb-core` depends on nothing but std + serde, everything else
depends on it, and `keyb-ui`/`keyb-lm-*` never depend on each other — they meet
only through channels owned by `keyb-app`.

| Crate | Contents |
|---|---|
| `keyb-core` | Geometry, steering dynamics, node tree, `LanguageModel` trait, smoothing. No I/O, no GPU. |
| `keyb-lm-llama` | llama.cpp backend, KV branch cache, speculative expansion worker. |
| `keyb-lm-ppm` | PPM fallback model and the persistent user-vocabulary layer. |
| `keyb-ui` | winit + layer-shell shells, vello scene building, input capture. |
| `keyb-inject` | Wayland virtual-keyboard/input-method and X11 XTEST output backends. |
| `keyb-app` | Binary: wiring, model provisioning, config. |

```bash
cargo test --workspace
```

Notable tests: `keyb-app/tests/oracle.rs` drives a simulated steerer with motor
noise against a WPM floor — that is the regression guard for dynamics changes;
`keyb-lm-llama/tests/real_model.rs` needs a downloaded model and skips without
one. `keyb-inject/tests/e2e_wayland.rs` requires `sway`, `swaymsg`, `foot`, and a
usable `XDG_RUNTIME_DIR`; it skips only when `sway` or `foot` is absent.

For UI work, `KEYB_SHELL=window KEYB_INJECT=buffer ./start.sh` gives a normal
window that writes nowhere.

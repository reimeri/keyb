# keyb

A mouse-driven, zooming on-screen keyboard in the lineage of
[Dasher](https://en.wikipedia.org/wiki/Dasher_(software)), with a local LLM as the
probability model. You write by steering through nested boxes whose sizes are the
model's predictive distribution — no clicking through keys. Committed text is
injected into whatever application has focus (Wayland or X11).

[SPEC.md](SPEC.md) is the design document and the source of truth for behaviour.

## Requirements

- Rust (stable) and a C/C++ toolchain, CMake, and libclang — `llama-cpp-2` builds
  llama.cpp from source and generates bindings with bindgen.
- A GPU with Vulkan/GL support for the vello renderer, and a sans-serif system font.
- Linux: Wayland (layer-shell) or X11. Other platforms are not supported yet.

## Run

```bash
./start.sh
```

That is `cargo run -p keyb-app --bin keyb`. On first run it downloads the default
model (Qwen2.5-0.5B-Instruct Q4_K_M, ~491 MB) to
`$XDG_DATA_HOME/keyb/models/`. If the download or load fails, keyb falls back to
the built-in PPM character model and still works.

Steering:

- On **Niri**, the keyboard follows the keyboard-focused application to its
  monitor when `NIRI_SOCKET` is available. Moving focus to an app on another
  output moves the keyboard there without resetting the current text or zoom
  state. If Niri IPC is temporarily unavailable, the keyboard stays on its
  last output while the connection retries in the background.
- **Hold the left button** to engage; release to stop.
- **Right of centre = write, left = back.** Backward zoom is the only undo, and
  zooming back across a commit boundary retracts already-injected text.
- **Vertical position** selects which box you drift toward.
- Scroll wheel or `+`/`-` adjusts speed; `Esc` quits. The on-screen buttons hide
  the keyboard and reset the current context.

Typed words are learned into a decaying personal vocabulary at
`$XDG_DATA_HOME/keyb/user_model.v1.bin`.

## Environment variables

| Variable | Effect |
|---|---|
| `KEYB_MODEL` | Path to a GGUF model instead of the managed default (disables auto-download). |
| `KEYB_NO_MODEL_DOWNLOAD` | Skip the default-model download; use PPM if the file is absent. |
| `KEYB_NO_PERSONALIZE` | Disable user-vocabulary learning. |
| `KEYB_ADAPTIVE` | Enable Dasher-style adaptive speed control (off by default). |
| `KEYB_SHELL` | `layer` \| `x11` \| `window` — window backend; auto-detected otherwise. `window` gives a plain resizable window, which is the convenient one for development. |
| `KEYB_INJECT` | `wayland` \| `wayland-vk` \| `x11` \| `buffer` — text output backend; `buffer` injects nothing. |
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
`keyb-lm-llama/tests/real_model.rs` needs a downloaded model, and
`keyb-inject/tests/e2e_wayland.rs` round-trips text through a headless `sway` +
`foot`. Both print a skip note when their dependencies are missing.

For UI work, `KEYB_SHELL=window KEYB_INJECT=buffer ./start.sh` gives a normal
window that writes nowhere.

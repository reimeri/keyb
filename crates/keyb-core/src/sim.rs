//! Oracle steerer (SPEC §11): a simulated user with motor noise and reaction
//! delay that steers optimally toward a target text. Its WPM is the
//! regression metric for dynamics/prediction changes.

use crate::engine::{Engine, PointerInput};

#[derive(Clone, Copy, Debug)]
pub struct OracleParams {
    /// Std-dev of Gaussian noise on the selection fraction.
    pub noise_y: f64,
    /// Reaction delay in whole frames (pointer lags intent).
    pub reaction_frames: usize,
    /// Simulation timestep (s).
    pub dt: f64,
    /// Give up after this much simulated time (s).
    pub max_secs: f64,
    /// How many symbols ahead the oracle aims (deeper = tighter target).
    pub lookahead: usize,
}

impl Default for OracleParams {
    fn default() -> Self {
        OracleParams {
            noise_y: 0.015,
            reaction_frames: 6, // ~100 ms at 60 fps
            dt: 1.0 / 60.0,
            max_secs: 600.0,
            // Aim deep: steering at the center of the *committing* box strands
            // the view at that box's center after the chain commit; aiming
            // several symbols in means the view lands on the next path.
            lookahead: 4,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OracleResult {
    pub completed: bool,
    pub chars: usize,
    pub secs: f64,
    pub wpm: f64,
}

/// Tiny deterministic RNG (xorshift64*) so the sim needs no dependencies.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box–Muller.
    fn gaussian(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

pub fn run_oracle(engine: &mut Engine, target: &str, p: &OracleParams, seed: u64) -> OracleResult {
    let target_syms = engine.alphabet.encode(target);
    assert!(!target_syms.is_empty(), "target text is empty after encoding");
    let mut rng = Rng(seed | 1);
    let mut delay_line: Vec<PointerInput> = Vec::new();
    let mut t = 0.0f64;
    // Two-mode controller with hysteresis. Advancing while the target point is
    // inside the view can never commit off-path text (any node containing the
    // whole view then also contains the target point, i.e. is a prefix of the
    // target). Zooming in while it is outside can — that, plus reaction delay,
    // turns bang-bang steering into a commit/retract limit cycle.
    let mut revealing = false;

    while t < p.max_secs {
        let committed = engine.committed_syms().to_vec();
        // Done?
        if committed == target_syms {
            break;
        }
        let intent = if !target_syms.starts_with(&committed) {
            // Wrote something wrong (noise) — back out until the prefix matches.
            revealing = false;
            PointerInput { dx: -1.0, y: 0.5, engaged: true }
        } else {
            let remaining = &target_syms[committed.len()..];
            let depth = p.lookahead.min(remaining.len());
            let (t_lo, t_hi) = engine.path_interval(&remaining[..depth]);
            let (v_lo, v_hi) = engine.view();
            let vw = v_hi - v_lo;
            let center = (t_lo + t_hi) / 2.0;
            let y_des = (center - v_lo) / vw;
            // Exit reveal as soon as the target point is inside the view:
            // advancing toward an in-view point cannot commit off-path text,
            // and demanding more depth is unsatisfiable when the target sits
            // near the current frame's edge.
            if revealing {
                if (0.12..=0.88).contains(&y_des) {
                    revealing = false;
                }
            } else if !(0.03..=0.97).contains(&y_des) {
                revealing = true;
            }
            if revealing {
                // Point at the edge nearest the target: the vertical pan
                // pulls it into view while a gentle zoom-out widens the view.
                PointerInput { dx: -0.3, y: y_des.clamp(0.0, 1.0), engaged: true }
            } else {
                PointerInput { dx: 1.0, y: y_des, engaged: true }
            }
        };

        // Reaction delay + motor noise.
        delay_line.push(intent);
        let idx = delay_line.len().saturating_sub(p.reaction_frames + 1);
        let mut input = delay_line[idx];
        if delay_line.len() > p.reaction_frames + 2 {
            delay_line.remove(0);
        }
        input.y += rng.gaussian() * p.noise_y;

        engine.step(input, p.dt);
        t += p.dt;
    }

    let chars = engine.committed_syms().len();
    let completed = engine.committed_syms() == target_syms.as_slice();
    let minutes = t / 60.0;
    OracleResult {
        completed,
        chars,
        secs: t,
        wpm: if minutes > 0.0 { (chars as f64 / 5.0) / minutes } else { 0.0 },
    }
}

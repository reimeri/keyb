/// A predictive model over the alphabet (SPEC §6.1).
///
/// M1 note: the interface is synchronous because PPM answers in microseconds.
/// The async LLM worker (M2) will sit behind an adapter that satisfies this
/// trait with cached/fallback distributions and upgrades in place.
pub trait LanguageModel: Send {
    /// P(next symbol | context). Must return exactly `alphabet_len` entries,
    /// every entry > 0, summing to ~1.
    fn predict(&mut self, context: &[u8]) -> Vec<f64>;

    /// Observe that `sym` followed `context` in committed text.
    fn learn(&mut self, _context: &[u8], _sym: u8) {}

    /// Remove a prior `learn(context, sym)` observation (the text was
    /// retracted). Called in reverse commit order with identical contexts, so
    /// implementations can restore their state exactly.
    fn unlearn(&mut self, _context: &[u8], _sym: u8) {}

    fn alphabet_len(&self) -> usize;
}

/// Uniform model — the floor every other model is smoothed against, and a
/// handy test double.
pub struct UniformModel(pub usize);

impl LanguageModel for UniformModel {
    fn predict(&mut self, _context: &[u8]) -> Vec<f64> {
        vec![1.0 / self.0 as f64; self.0]
    }

    fn alphabet_len(&self) -> usize {
        self.0
    }
}

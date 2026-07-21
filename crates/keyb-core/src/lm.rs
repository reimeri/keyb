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

    /// Signal that `sym` after `context` was genuinely committed by the user
    /// (paired with a `TextEvent::Commit`), as opposed to the reversible
    /// `learn` bookkeeping. Unlike `learn`/`unlearn`, this is *not* emitted by
    /// `Engine::reset`, so persistent personalization layers can key long-term
    /// user-vocabulary adaptation off it and survive resets.
    fn note_commit(&mut self, _context: &[u8], _sym: u8) {}

    /// Signal that a previously `note_commit`ted `sym` was retracted by the
    /// user (paired with a `TextEvent::Retract`). Reset never emits this.
    fn note_retract(&mut self, _context: &[u8], _sym: u8) {}

    /// Distributions that arrived asynchronously since the last call, as
    /// (context, distribution) pairs. `predict` may answer with a fallback
    /// (SPEC §6.3); when the real distribution lands, the engine re-partitions
    /// the matching node and animates sizes so boxes never pop. Synchronous
    /// models never upgrade.
    fn poll_upgrades(&mut self) -> Vec<(Vec<u8>, Vec<f64>)> {
        Vec::new()
    }

    /// Once-per-step hint: the context of the deepest expanded node under the
    /// view center. Async backends re-prioritize pending work toward it and
    /// speculatively expand its likely children (SPEC §6.6).
    fn frame_hint(&mut self, _center_context: &[u8]) {}

    /// The node for `context` was evicted; pending work for it may be
    /// cancelled and any cache branch freed (SPEC §6.1, §6.5).
    fn release(&mut self, _context: &[u8]) {}

    fn alphabet_len(&self) -> usize;
}

/// Boxes forward, so wrappers can be generic over owned or boxed models.
impl<L: LanguageModel + ?Sized> LanguageModel for Box<L> {
    fn predict(&mut self, context: &[u8]) -> Vec<f64> {
        (**self).predict(context)
    }

    fn learn(&mut self, context: &[u8], sym: u8) {
        (**self).learn(context, sym);
    }

    fn unlearn(&mut self, context: &[u8], sym: u8) {
        (**self).unlearn(context, sym);
    }

    fn note_commit(&mut self, context: &[u8], sym: u8) {
        (**self).note_commit(context, sym);
    }

    fn note_retract(&mut self, context: &[u8], sym: u8) {
        (**self).note_retract(context, sym);
    }

    fn poll_upgrades(&mut self) -> Vec<(Vec<u8>, Vec<f64>)> {
        (**self).poll_upgrades()
    }

    fn frame_hint(&mut self, center_context: &[u8]) {
        (**self).frame_hint(center_context);
    }

    fn release(&mut self, context: &[u8]) {
        (**self).release(context);
    }

    fn alphabet_len(&self) -> usize {
        (**self).alphabet_len()
    }
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

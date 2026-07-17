//! Order-N PPM character model (SPEC §6.3 fallback / M1 primary model).
//!
//! Blending follows the Dasher lineage: for each context order k (ascending),
//! seen symbols contribute discounted counts and unseen mass escapes to the
//! shorter context, bottoming out at uniform. Learns online from commits.

use std::collections::HashMap;

use keyb_core::{Alphabet, LanguageModel};

const SEED_CORPUS: &str = include_str!("../assets/seed.txt");

struct Counts {
    counts: Vec<u32>,
    total: u64,
    distinct: u32,
}

pub struct PpmModel {
    order: usize,
    n_sym: usize,
    /// Escape smoothing constants (Dasher uses ~0.49/0.77).
    alpha: f64,
    beta: f64,
    /// Probability floor per symbol after blending (SPEC §6.3).
    p_min: f64,
    contexts: HashMap<Box<[u8]>, Counts>,
}

impl PpmModel {
    pub fn new(n_sym: usize, order: usize) -> Self {
        PpmModel {
            order,
            n_sym,
            alpha: 0.49,
            beta: 0.77,
            p_min: 1e-4,
            contexts: HashMap::new(),
        }
    }

    /// Order-4 model pre-trained on the embedded seed corpus.
    pub fn seeded(alphabet: &Alphabet) -> Self {
        let mut m = PpmModel::new(alphabet.len(), 4);
        m.train_text(alphabet, SEED_CORPUS);
        m
    }

    pub fn train_text(&mut self, alphabet: &Alphabet, text: &str) {
        let syms = alphabet.encode(text);
        for i in 0..syms.len() {
            self.learn_at(&syms[..i], syms[i]);
        }
    }

    fn learn_at(&mut self, context: &[u8], sym: u8) {
        for k in 0..=self.order.min(context.len()) {
            let key = &context[context.len() - k..];
            let entry = self.contexts.entry(key.into()).or_insert_with(|| Counts {
                counts: vec![0; self.n_sym],
                total: 0,
                distinct: 0,
            });
            if entry.counts[sym as usize] == 0 {
                entry.distinct += 1;
            }
            entry.counts[sym as usize] += 1;
            entry.total += 1;
        }
    }
}

impl LanguageModel for PpmModel {
    fn predict(&mut self, context: &[u8]) -> Vec<f64> {
        let n = self.n_sym;
        // Order -1: uniform.
        let mut p = vec![1.0 / n as f64; n];
        // Ascend through orders; each seen context re-blends over the shorter one.
        for k in 0..=self.order.min(context.len()) {
            let key = &context[context.len() - k..];
            let Some(c) = self.contexts.get(key) else {
                continue;
            };
            if c.total == 0 {
                continue;
            }
            let denom = c.total as f64 + self.alpha;
            let escape = ((self.alpha + self.beta * c.distinct as f64) / denom).clamp(0.0, 1.0);
            for s in 0..n {
                let cnt = c.counts[s] as f64;
                let direct = if cnt > 0.0 {
                    (cnt - self.beta).max(0.0) / denom
                } else {
                    0.0
                };
                p[s] = direct + escape * p[s];
            }
        }
        // Floor + renormalize: every symbol must stay reachable.
        let mut sum = 0.0;
        for v in p.iter_mut() {
            *v = v.max(self.p_min);
            sum += *v;
        }
        for v in p.iter_mut() {
            *v /= sum;
        }
        p
    }

    fn learn(&mut self, context: &[u8], sym: u8) {
        self.learn_at(context, sym);
    }

    /// Exact inverse of `learn`: retracted text must leave no trace, so
    /// re-expanded boxes match what was on screen before the commit.
    fn unlearn(&mut self, context: &[u8], sym: u8) {
        for k in 0..=self.order.min(context.len()) {
            let key = &context[context.len() - k..];
            if let Some(entry) = self.contexts.get_mut(key) {
                if entry.counts[sym as usize] > 0 {
                    entry.counts[sym as usize] -= 1;
                    entry.total -= 1;
                    if entry.counts[sym as usize] == 0 {
                        entry.distinct -= 1;
                    }
                }
            }
        }
    }

    fn alphabet_len(&self) -> usize {
        self.n_sym
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_is_normalized_and_positive() {
        let a = Alphabet::english_v1();
        let mut m = PpmModel::seeded(&a);
        for ctx in [&b""[..], &a.encode("the "), &a.encode("qzx")] {
            let p = m.predict(ctx);
            assert_eq!(p.len(), a.len());
            let sum: f64 = p.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "sum={sum}");
            assert!(p.iter().all(|&v| v > 0.0));
        }
    }

    #[test]
    fn seed_corpus_learns_english_bigrams() {
        let a = Alphabet::english_v1();
        let mut m = PpmModel::seeded(&a);
        // After "th", 'e' should be far more likely than 'q'.
        let ctx = a.encode("th");
        let p = m.predict(&ctx);
        let e = p[a.index_of('e').unwrap() as usize];
        let q = p[a.index_of('q').unwrap() as usize];
        assert!(e > 10.0 * q, "e={e} q={q}");
    }

    #[test]
    fn unlearn_restores_distribution_exactly() {
        let a = Alphabet::english_v1();
        let mut m = PpmModel::seeded(&a);
        let probes = [a.encode(""), a.encode("th"), a.encode("the weather")];
        let before: Vec<Vec<f64>> = probes.iter().map(|c| m.predict(c)).collect();
        // Mirror engine usage: learn against the growing committed text,
        // unlearn in reverse.
        let mut committed = a.encode("mistyped junk");
        let typed = a.encode("qzq qzq");
        for &s in &typed {
            m.learn(&committed, s);
            committed.push(s);
        }
        for _ in 0..typed.len() {
            let s = committed.pop().unwrap();
            m.unlearn(&committed, s);
        }
        let after: Vec<Vec<f64>> = probes.iter().map(|c| m.predict(c)).collect();
        assert_eq!(before, after);
    }

    #[test]
    fn online_learning_shifts_mass() {
        let a = Alphabet::english_v1();
        let mut m = PpmModel::new(a.len(), 4);
        let ctx = a.encode("zz");
        let before = m.predict(&ctx)[a.index_of('y').unwrap() as usize];
        for _ in 0..50 {
            m.learn(&ctx, a.index_of('y').unwrap());
        }
        let after = m.predict(&ctx)[a.index_of('y').unwrap() as usize];
        assert!(after > before * 5.0, "before={before} after={after}");
    }
}

//! Token → character marginalization (SPEC §6.2), pure and model-free.
//!
//! The LLM predicts subword tokens; the UI is character-honest. Given a
//! next-token distribution taken at a token boundary and the characters the
//! user has spelled since that boundary (the intra-token prefix `s`, token
//! healing), this module answers: how much token mass continues with each
//! alphabet character, and how much ends exactly at `s` (whose continuation
//! needs the next branch's distribution).

use keyb_core::Alphabet;

/// Precomputed view of the vocabulary in alphabet-symbol space.
pub struct VocabIndex {
    /// Per token: its bytes mapped to alphabet symbols, truncated at the
    /// first byte outside the alphabet (uppercase, UTF-8 tails, specials).
    tok_syms: Vec<Box<[u8]>>,
    /// Every byte of the token is inside the alphabet (and there is at least
    /// one). Only such tokens can be *exactly* consumed by a steered path.
    tok_full: Vec<bool>,
    /// Token ids with non-empty `tok_syms`, sorted by `tok_syms`, so all
    /// tokens sharing a symbol prefix form one contiguous range.
    sorted: Vec<u32>,
    n_sym: usize,
}

impl VocabIndex {
    pub fn new(pieces: impl Iterator<Item = Vec<u8>>, alphabet: &Alphabet) -> Self {
        let mut tok_syms = Vec::new();
        let mut tok_full = Vec::new();
        for bytes in pieces {
            let mut syms = Vec::new();
            let mut full = !bytes.is_empty();
            for &b in &bytes {
                match alphabet.index_of(b as char) {
                    Some(s) if b.is_ascii() => syms.push(s),
                    _ => {
                        full = false;
                        break;
                    }
                }
            }
            tok_syms.push(syms.into_boxed_slice());
            tok_full.push(full);
        }
        let mut sorted: Vec<u32> = (0..tok_syms.len() as u32)
            .filter(|&t| !tok_syms[t as usize].is_empty())
            .collect();
        sorted.sort_by(|&a, &b| tok_syms[a as usize].cmp(&tok_syms[b as usize]));
        VocabIndex {
            tok_syms,
            tok_full,
            sorted,
            n_sym: alphabet.len(),
        }
    }

    /// Symbols of token `t` (truncated at the first out-of-alphabet byte).
    pub fn syms_of(&self, t: u32) -> &[u8] {
        &self.tok_syms[t as usize]
    }

    /// Contiguous `sorted` range of tokens whose symbols start with `s`.
    fn prefix_range(&self, s: &[u8]) -> (usize, usize) {
        let lo = self
            .sorted
            .partition_point(|&t| self.tok_syms[t as usize].as_ref() < s);
        let hi = self.sorted.partition_point(|&t| {
            let ts = &self.tok_syms[t as usize];
            let k = ts.len().min(s.len());
            ts[..k] <= s[..k]
        });
        (lo, hi)
    }

    /// Marginalize `probs` (a next-token distribution, indexed by token id)
    /// against intra-token prefix `s`. Returns per-symbol mass of tokens
    /// continuing past `s` with that symbol, plus the mass of tokens ending
    /// exactly at `s` (their continuation lives in the next branch).
    /// Token mass outside the alphabet is dropped — the caller renormalizes.
    pub fn char_masses(&self, probs: &[f32], s: &[u8]) -> (Vec<f64>, f64) {
        let mut cont = vec![0.0f64; self.n_sym];
        let mut exact = 0.0f64;
        let (lo, hi) = self.prefix_range(s);
        for &t in &self.sorted[lo..hi] {
            let ts = &self.tok_syms[t as usize];
            let p = probs[t as usize] as f64;
            if ts.len() > s.len() {
                cont[ts[s.len()] as usize] += p;
            } else if self.tok_full[t as usize] {
                // ts == s by range membership; a token with trailing
                // out-of-alphabet bytes can never end a steered path.
                exact += p;
            }
        }
        (cont, exact)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(pieces: &[&str]) -> (VocabIndex, Alphabet) {
        let a = Alphabet::english_v1();
        let idx = VocabIndex::new(pieces.iter().map(|p| p.as_bytes().to_vec()), &a);
        (idx, a)
    }

    #[test]
    fn masses_split_between_continuations_and_exact() {
        // Vocab: "str", "straw", "strawberry", "street", "Straw" (case-invalid),
        // "str\xff" (invalid tail).
        let (v, a) = vocab(&[
            "str",
            "straw",
            "strawberry",
            "street",
            "Straw",
            "str\u{00ff}",
        ]);
        let probs = [0.1f32, 0.2, 0.3, 0.25, 0.1, 0.05];
        let s = a.encode("str");
        let (cont, exact) = v.char_masses(&probs, &s);
        // Continuations past "str": "straw"/"strawberry" contribute to 'a',
        // "street" to 'e'. "Straw" doesn't match (case). "str\xff" has valid
        // syms exactly "str" but is not fully valid → contributes nowhere.
        assert!((cont[a.index_of('a').unwrap() as usize] - 0.5).abs() < 1e-6);
        assert!((cont[a.index_of('e').unwrap() as usize] - 0.25).abs() < 1e-6);
        assert!(
            (exact - 0.1).abs() < 1e-6,
            "only exact 'str' ends here: {exact}"
        );
        let total: f64 = cont.iter().sum::<f64>() + exact;
        assert!((total - 0.85).abs() < 1e-6, "invalid-mass dropped: {total}");
    }

    #[test]
    fn empty_prefix_marginalizes_first_chars() {
        let (v, a) = vocab(&["the", "then", "cat", "<eos>", " the"]);
        let probs = [0.4f32, 0.1, 0.2, 0.25, 0.05];
        let (cont, exact) = v.char_masses(&probs, &[]);
        assert!((cont[a.index_of('t').unwrap() as usize] - 0.5).abs() < 1e-6);
        assert!((cont[a.index_of('c').unwrap() as usize] - 0.2).abs() < 1e-6);
        assert!((cont[a.index_of(' ').unwrap() as usize] - 0.05).abs() < 1e-6);
        assert_eq!(exact, 0.0);
        // "<eos>" starts with '<' — outside the alphabet, dropped.
        let total: f64 = cont.iter().sum();
        assert!((total - 0.75).abs() < 1e-6);
    }

    /// Exhaustive cross-check (SPEC §11): trie masses equal summed token
    /// masses for random-ish vocab/prefix combinations.
    #[test]
    fn char_masses_equal_bruteforce_sums() {
        let pieces = [
            "a",
            "ab",
            "abc",
            "b",
            "ba",
            " ",
            " a",
            "a b",
            "'",
            "a'",
            "ab'",
            "A",
            "aB",
            "ab\u{00e9}",
            "",
            "abcd",
            "abcde",
        ];
        let (v, a) = vocab(&pieces);
        let mut probs = vec![0.0f32; pieces.len()];
        for (i, p) in probs.iter_mut().enumerate() {
            *p = 1.0 / (i as f32 + 2.0);
        }
        for s_text in ["", "a", "ab", "abc", "b", " ", "z"] {
            let s = a.encode(s_text);
            let (cont, exact) = v.char_masses(&probs, &s);
            // Brute force over the raw pieces.
            let mut b_cont = vec![0.0f64; a.len()];
            let mut b_exact = 0.0f64;
            for (t, piece) in pieces.iter().enumerate() {
                let syms: Vec<u8> = piece
                    .chars()
                    .map_while(|c| if c.is_ascii() { a.index_of(c) } else { None })
                    .collect();
                let valid = piece
                    .chars()
                    .all(|c| c.is_ascii() && a.index_of(c).is_some())
                    && !piece.is_empty();
                if syms.len() > s.len() && syms.starts_with(&s) {
                    b_cont[syms[s.len()] as usize] += probs[t] as f64;
                } else if valid && syms == s {
                    b_exact += probs[t] as f64;
                }
            }
            for (x, y) in cont.iter().zip(&b_cont) {
                assert!((x - y).abs() < 1e-9, "prefix {s_text:?}: {x} vs {y}");
            }
            assert!((exact - b_exact).abs() < 1e-9, "prefix {s_text:?}");
        }
    }
}

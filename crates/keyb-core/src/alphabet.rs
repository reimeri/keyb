/// Fixed, ordered set of writable symbols.
///
/// The ordering is part of the user's spatial memory and must never change at
/// runtime (SPEC §5.4). Sizes change with probability; positions in the column
/// order do not.
#[derive(Clone, Debug)]
pub struct Alphabet {
    chars: Vec<char>,
}

impl Alphabet {
    /// v1 English alphabet: a–z, space, then a small punctuation tail.
    pub fn english_v1() -> Self {
        let mut chars: Vec<char> = ('a'..='z').collect();
        chars.push(' ');
        chars.extend(['.', ',', '\'']);
        Alphabet { chars }
    }

    pub fn len(&self) -> usize {
        self.chars.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    pub fn char_of(&self, sym: u8) -> char {
        self.chars[sym as usize]
    }

    pub fn index_of(&self, c: char) -> Option<u8> {
        self.chars.iter().position(|&x| x == c).map(|i| i as u8)
    }

    /// Display label for a symbol (space gets a visible mark).
    pub fn label(&self, sym: u8) -> char {
        match self.char_of(sym) {
            ' ' => '␣',
            c => c,
        }
    }

    /// Map a string to symbol indices, lowercasing and skipping anything
    /// outside the alphabet.
    pub fn encode(&self, text: &str) -> Vec<u8> {
        text.chars()
            .flat_map(|c| c.to_lowercase())
            .filter_map(|c| self.index_of(c))
            .collect()
    }

    pub fn decode(&self, syms: &[u8]) -> String {
        syms.iter().map(|&s| self.char_of(s)).collect()
    }
}

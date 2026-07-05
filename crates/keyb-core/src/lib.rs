//! keyb-core: pure model of the zooming text-entry interface.
//!
//! No I/O, no GPU, no clocks — everything is a deterministic function of
//! (input trace, language model). See SPEC.md §5.

pub mod alphabet;
pub mod engine;
pub mod lm;
pub mod sim;

pub use alphabet::Alphabet;
pub use engine::{BoxView, Engine, EngineParams, PointerInput};
pub use lm::LanguageModel;

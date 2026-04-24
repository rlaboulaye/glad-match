pub mod candidates;
pub mod cost;
pub mod error;
pub mod features;
pub mod filter;
pub mod io;
pub mod ot;
pub mod pipeline;
pub mod refine;
pub mod stats;

pub use pipeline::{MatchParams, MatchResult, match_controls};

pub use error::Error;

#[cfg(test)]
mod integration_tests;

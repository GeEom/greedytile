//! Test harness organizing unit and meta-level validation

use greedytile::*;

/// What a test returns, so a failed step reports its error instead of panicking
pub type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

mod meta;
mod unit;

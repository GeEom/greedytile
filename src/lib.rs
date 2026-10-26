//! Grows new images from the 3x3 tiles of a source image, placing them where
//! the fewest tiles fit and matching the source's colour ratios and spacing

#![forbid(unsafe_code)]

/// Core algorithm implementation including tile selection, propagation, and deadlock resolution
pub mod algorithm;
/// Statistical analysis and pattern preprocessing for source images
pub mod analysis;
/// Input/output operations and error handling
pub mod io;
/// Mathematical utilities for interpolation and probability calculations
pub mod math;
/// Spatial grid management and tile extraction utilities
pub mod spatial;

pub use io::error::{AlgorithmError, Result};

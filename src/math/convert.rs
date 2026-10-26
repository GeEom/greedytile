//! Each helper returns what an `as` cast does for every value the algorithm
//! produces, and names what it does outside that range

use num_traits::ToPrimitive;

/// Converts a count or index to `f64`, rounding past 2^53 as `as` does
#[must_use]
pub fn usize_to_f64(value: usize) -> f64 {
    value.to_f64().unwrap_or(f64::MAX)
}

/// Converts a 64-bit count to `f64`, rounding past 2^53 as `as` does
#[must_use]
pub fn u64_to_f64(value: u64) -> f64 {
    value.to_f64().unwrap_or(f64::MAX)
}

/// Converts a count or index to `i32`, saturating at `i32::MAX`
#[must_use]
pub fn usize_to_i32(value: usize) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}

/// Converts a count or index to `u32`, saturating at `u32::MAX`
#[must_use]
pub fn usize_to_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Truncates toward zero into a `u8`, clamping to 0..=255 with NaN at 0, as `as` does
#[must_use]
pub fn f64_to_u8_saturating(value: f64) -> u8 {
    value
        .to_u8()
        .unwrap_or(if value > 0.0 { u8::MAX } else { 0 })
}

/// Truncates toward zero into a `usize`, clamping with NaN at 0, as `as` does
#[must_use]
pub fn f64_to_usize_saturating(value: f64) -> usize {
    value
        .to_usize()
        .unwrap_or(if value > 0.0 { usize::MAX } else { 0 })
}

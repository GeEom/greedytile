//! Tests that each conversion matches an `as` cast in range and clamps outside it

#[cfg(test)]
mod tests {
    use greedytile::math::convert::{
        f64_to_u8_saturating, f64_to_usize_saturating, u64_to_f64, usize_to_f64, usize_to_i32,
        usize_to_u32,
    };

    // Verified by rounding instead of truncating in f64_to_u8_saturating
    #[test]
    fn test_f64_to_u8_truncates_and_clamps() {
        assert_eq!(f64_to_u8_saturating(12.7), 12);
        assert_eq!(f64_to_u8_saturating(255.9), 255);
        assert_eq!(f64_to_u8_saturating(256.0), 255);
        assert_eq!(f64_to_u8_saturating(f64::INFINITY), 255);
        assert_eq!(f64_to_u8_saturating(-0.5), 0);
        assert_eq!(f64_to_u8_saturating(-3.0), 0);
        assert_eq!(f64_to_u8_saturating(f64::NAN), 0);
    }

    // Verified by returning 0 for every out-of-range value
    #[test]
    fn test_f64_to_usize_truncates_and_clamps() {
        assert_eq!(f64_to_usize_saturating(7.99), 7);
        assert_eq!(f64_to_usize_saturating(1e30), usize::MAX);
        assert_eq!(f64_to_usize_saturating(-1.5), 0);
        assert_eq!(f64_to_usize_saturating(f64::NAN), 0);
    }

    // 2^53 + 1 has no f64, and rounds to the even neighbour 2^53
    #[test]
    fn test_integer_to_f64_rounds_past_mantissa() {
        assert!((usize_to_f64(3) - 3.0).abs() < f64::EPSILON);
        assert!((u64_to_f64(9_007_199_254_740_993) - 9_007_199_254_740_992.0).abs() < 1.0);
        assert!((usize_to_f64(usize::MAX) - 18_446_744_073_709_551_616.0).abs() < 1.0);
    }

    // Verified by wrapping instead of saturating
    #[test]
    fn test_usize_to_narrow_integers_saturate() {
        assert_eq!(usize_to_i32(42), 42);
        assert_eq!(usize_to_i32(usize::MAX), i32::MAX);
        assert_eq!(usize_to_u32(42), 42);
        assert_eq!(usize_to_u32(usize::MAX), u32::MAX);
    }
}

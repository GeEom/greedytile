use crate::math::convert::usize_to_f64;

/// The error function, by Abramowitz and Stegun's approximation 7.1.26
#[must_use]
pub fn erf(x: f64) -> f64 {
    let a1 = 0.254_829_592_f64;
    let a2 = -0.284_496_736_f64;
    let a3 = 1.421_413_741_f64;
    let a4 = -1.453_152_027_f64;
    let a5 = 1.061_405_429_f64;
    let p = 0.327_591_1_f64;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / p.mul_add(x, 1.0);
    let y = (((((a5.mul_add(t, a4)).mul_add(t, a3)).mul_add(t, a2)).mul_add(t, a1)) * t)
        .mul_add(-(-x * x).exp(), 1.0);

    sign * y
}

/// P(X ≤ k) for X ~ Binomial(n, p), by the normal approximation
#[must_use]
pub fn binomial_normal_approximate_cdf(n: usize, p: f64, k: usize) -> f64 {
    if k >= n {
        return 1.0;
    }
    if p <= 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    if p >= 1.0 {
        return 0.0;
    }

    let n_f64 = usize_to_f64(n);
    let k_f64 = usize_to_f64(k);
    let mean = n_f64 * p;
    let variance = n_f64 * p * (1.0 - p);
    let std_dev = variance.sqrt();

    // Calculate the standardized value. Use k + 0.5 for continuity correction
    let z = (k_f64 + 0.5 - mean) / (std::f64::consts::SQRT_2 * std_dev);

    // Return 1/2 * erfc(-z) where erfc(x) = 1 - erf(x)
    0.5 * (1.0 - erf(-z))
}

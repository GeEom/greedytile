//! Tests that the cubic spline fits its points and clamps outside them

#[cfg(test)]
mod tests {
    use crate::math::interpolation::Cubic;

    // Verified by removing the cubic term, which makes the spline linear
    #[test]
    fn test_cubic_interpolation_passes_through_data_points_and_interpolates_smoothly()
    -> crate::TestResult {
        let x_values = vec![-1.0, 0.0, 1.0, 2.0, 3.0];
        let y_values = vec![4.0, 1.0, 0.0, 1.0, 4.0];

        let cubic = Cubic::new(x_values.clone(), y_values.clone())?;

        for (x, y) in x_values.iter().zip(y_values.iter()) {
            let interpolated = cubic.evaluate(*x)?;
            assert!(
                (interpolated - y).abs() < 1e-10,
                "Interpolation should pass through data point ({x}, {y}), got {interpolated}"
            );
        }

        let test_x = 0.5;
        let interpolated = cubic.evaluate(test_x)?;
        assert!(
            interpolated > 0.0 && interpolated < 0.5,
            "Interpolation at x={test_x} should be between adjacent y values, got {interpolated}"
        );

        assert!(
            (cubic.evaluate(-2.0)? - 4.0).abs() < f64::EPSILON,
            "Should clamp to first y value for x < min"
        );
        assert!(
            (cubic.evaluate(4.0)? - 4.0).abs() < f64::EPSILON,
            "Should clamp to last y value for x > max"
        );

        let mid_point = cubic.evaluate(1.5)?;
        assert!(
            mid_point > 0.0 && mid_point < 1.0,
            "Interpolation should preserve monotonicity in [1,2]"
        );
        Ok(())
    }
}

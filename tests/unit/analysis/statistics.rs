//! Tests for smooth kernel distribution with boundary reflection

#[cfg(test)]
mod tests {
    use crate::analysis::statistics::SmoothKernelDistribution;

    // Tests bounded kernel density estimation with reflection at x=0 using mathematical verification and measurable reflection effects
    // Verified by mathematical verification of reflection implementation with exact expected values
    #[test]
    fn test_smooth_kernel_distribution_pdf_with_reflection() {
        let weighted_data = vec![(2.0, 1.0), (5.0, 2.0), (8.0, 1.0)];

        let dist = SmoothKernelDistribution::new((0, 1), weighted_data);

        assert!((dist.pdf(-1.0) - 0.0).abs() < f64::EPSILON);
        assert!((dist.pdf(-0.1) - 0.0).abs() < f64::EPSILON);

        let pdf_at_zero = dist.pdf(0.0);
        assert!(
            pdf_at_zero > 0.0,
            "PDF at boundary should be positive due to reflection"
        );

        let pdf_near_point = dist.pdf(2.1);
        let pdf_far_from_points = dist.pdf(15.0);
        assert!(
            pdf_near_point > pdf_far_from_points,
            "PDF should be higher near data points than far away"
        );

        let pdf_at_low_weight = dist.pdf(2.0);
        let pdf_at_high_weight = dist.pdf(5.0);
        assert!(
            pdf_at_high_weight > pdf_at_low_weight,
            "PDF should be higher at points with higher weights"
        );

        let simple_data = vec![(1.0, 1.0)];
        let simple_dist = SmoothKernelDistribution::new((0, 1), simple_data);

        let x_test: f64 = 0.5;
        let h: f64 = 1.0;
        let total_weight: f64 = 1.0;
        let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();

        let normal_contribution = ((-0.5_f64 * ((x_test - 1.0) / h).powi(2)).exp()) / sqrt_2pi;
        let reflected_contribution = ((-0.5_f64 * ((x_test + 1.0) / h).powi(2)).exp()) / sqrt_2pi;
        let expected_pdf = (normal_contribution + reflected_contribution) / total_weight;

        let actual_pdf = simple_dist.pdf(x_test);
        assert!(
            (actual_pdf - expected_pdf).abs() < 1e-10,
            "PDF at x={x_test} should match mathematical expectation. Expected: {expected_pdf}, Got: {actual_pdf}"
        );

        let pdf_near_boundary = simple_dist.pdf(0.1);
        let pdf_far_side = simple_dist.pdf(1.9);

        assert!(
            pdf_near_boundary > pdf_far_side,
            "Reflection should make PDF higher near x=0 boundary. PDF(0.1)={pdf_near_boundary}, PDF(1.9)={pdf_far_side}"
        );

        let x1: f64 = 0.1;
        let x2: f64 = 1.9;

        let normal_01 = ((-0.5_f64 * ((x1 - 1.0) / h).powi(2)).exp()) / sqrt_2pi;
        let reflected_01 = ((-0.5_f64 * ((x1 + 1.0) / h).powi(2)).exp()) / sqrt_2pi;
        let expected_01 = (normal_01 + reflected_01) / total_weight;

        let normal_19 = ((-0.5_f64 * ((x2 - 1.0) / h).powi(2)).exp()) / sqrt_2pi;
        let reflected_19 = ((-0.5_f64 * ((x2 + 1.0) / h).powi(2)).exp()) / sqrt_2pi;
        let expected_19 = (normal_19 + reflected_19) / total_weight;

        assert!(
            (simple_dist.pdf(x1) - expected_01).abs() < 1e-10,
            "PDF calculation at x=0.1 should match manual calculation"
        );
        assert!(
            (simple_dist.pdf(x2) - expected_19).abs() < 1e-10,
            "PDF calculation at x=1.9 should match manual calculation"
        );

        let reflection_effect = reflected_01;
        assert!(
            reflection_effect > 1e-3,
            "Reflection should have a measurable effect at x=0.1, contribution: {reflection_effect}"
        );
    }

    // Ensures PDF returns exactly 0.0 for negative values
    // Verified by testing boundary check for negative values in PDF
    #[test]
    fn test_pdf_negative_values_strict() {
        let weighted_data = vec![(0.1, 1.0), (0.5, 1.0), (1.0, 1.0)];

        let dist = SmoothKernelDistribution::new((0, 1), weighted_data);

        let negative_test_values = vec![-0.001, -0.01, -0.1, -0.5, -1.0, -2.0, -10.0, -100.0];

        for &x in &negative_test_values {
            let pdf_value = dist.pdf(x);
            assert!(
                pdf_value.abs() < f64::EPSILON,
                "PDF must return exactly 0.0 for negative x={x}, but got {pdf_value}"
            );
        }

        assert!(
            dist.pdf(-f64::EPSILON).abs() < f64::EPSILON,
            "PDF at -epsilon must be 0.0"
        );
        assert!(
            dist.pdf(-1e-15).abs() < f64::EPSILON,
            "PDF at very small negative must be 0.0"
        );
        assert!(
            dist.pdf(-1e-10).abs() < f64::EPSILON,
            "PDF at small negative must be 0.0"
        );

        assert!(dist.pdf(0.0) > 0.0, "PDF at x=0 should be positive");
        assert!(
            dist.pdf(f64::EPSILON) > 0.0,
            "PDF at +epsilon should be positive"
        );
        assert!(
            dist.pdf(1e-10) > 0.0,
            "PDF at small positive should be positive"
        );
    }

    // Tests that negative values return exactly 0.0 using strict equality checks and comprehensive boundary testing
    // Verified by using strict equality to verify negative values must return exactly 0.0
    #[test]
    #[allow(clippy::float_cmp)]
    fn test_pdf_without_negative_check_would_fail() {
        let weighted_data = vec![(0.1, 1.0)];
        let dist = SmoothKernelDistribution::new((0, 1), weighted_data);

        let pdf_at_negative = dist.pdf(-0.05);
        assert_eq!(
            pdf_at_negative, 0.0,
            "PDF at x=-0.05 must be exactly 0.0, not {pdf_at_negative} (would be non-zero without boundary check)"
        );

        let pdf_at_tiny_negative = dist.pdf(-1e-100);
        assert_eq!(
            pdf_at_tiny_negative, 0.0,
            "PDF at x=-1e-100 must be exactly 0.0"
        );

        let test_negative_values = [-0.001, -0.1, -1.0, -10.0, -f64::MIN_POSITIVE];
        for &neg_x in &test_negative_values {
            let pdf_val = dist.pdf(neg_x);
            assert_eq!(
                pdf_val, 0.0,
                "PDF at x={neg_x} must be exactly 0.0, got {pdf_val}"
            );
        }

        let pdf_at_zero = dist.pdf(0.0);
        assert!(
            pdf_at_zero > 0.0,
            "PDF at x=0.0 should be positive, got {pdf_at_zero}"
        );

        let pdf_at_positive = dist.pdf(0.1);
        assert!(
            pdf_at_positive > 0.0,
            "PDF at x=0.1 should be positive, got {pdf_at_positive}"
        );
    }
}

//! Tests for smooth kernel distribution with boundary reflection

#[cfg(test)]
mod tests {
    use crate::analysis::statistics::SmoothKernelDistribution;
    use greedytile::analysis::statistics::Processor;
    use greedytile::spatial::tiles::TileExtractor;
    use ndarray::{Array2, Array4};

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
    fn test_pdf_without_negative_check_would_fail() {
        let weighted_data = vec![(0.1, 1.0)];
        let dist = SmoothKernelDistribution::new((0, 1), weighted_data);

        let pdf_at_negative = dist.pdf(-0.05);
        // Comparing bits demands exactly +0.0
        assert_eq!(
            pdf_at_negative.to_bits(),
            0.0_f64.to_bits(),
            "PDF at x=-0.05 must be exactly 0.0, not {pdf_at_negative} (would be non-zero without boundary check)"
        );

        let pdf_at_tiny_negative = dist.pdf(-1e-100);
        assert_eq!(
            pdf_at_tiny_negative.to_bits(),
            0.0_f64.to_bits(),
            "PDF at x=-1e-100 must be exactly 0.0"
        );

        let test_negative_values = [-0.001, -0.1, -1.0, -10.0, -f64::MIN_POSITIVE];
        for &neg_x in &test_negative_values {
            let pdf_val = dist.pdf(neg_x);
            assert_eq!(
                pdf_val.to_bits(),
                0.0_f64.to_bits(),
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

    // Tests that tiles beyond the influence distance leave every colour's odds alone
    // Verified by the far field fading to the source ratio, which recounts the prior per neighbour
    #[test]
    fn test_far_field_influence_is_neutral() {
        // Vertical stripes, two of colour 1 to one of colour 2
        let source = Array2::from_shape_fn((9, 9), |(_, j)| if j % 3 == 2 { 2 } else { 1 });
        let processor = Processor::new(source, vec![2.0 / 3.0, 1.0 / 3.0], 6, 6);
        let sample_points = TileExtractor::calculate_exponential_sample_points(6.0);

        let matrices = processor
            .preprocess_pattern_statistics(&sample_points)
            .unwrap_or_else(|_| Array4::zeros((2, 2, 13, 13)));

        let at = |index: [usize; 4]| matrices.get(index).copied().unwrap_or(f64::NAN);
        for from in 0..2 {
            // The corner is ~8.5 away, the cell beside the centre 1 away
            let far = [at([from, 0, 0, 0]), at([from, 1, 0, 0])];
            let near = [at([from, 0, 6, 7]), at([from, 1, 6, 7])];
            assert!(far[0] > 0.0, "Far influence should be positive");
            assert!(
                (far[0] / far[1] - 1.0).abs() < 1e-12,
                "Far influence should not favour a colour, got {far:?}"
            );
            assert!(
                (near[0] / near[1] - 1.0).abs() > 1e-3,
                "Near influence should still favour a colour, got {near:?}"
            );
        }
    }
}

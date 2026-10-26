//! Tests for image pattern analysis and color ratio calculations

#[cfg(test)]
mod tests {
    use greedytile::analysis::patterns::ImageProcessor;
    use greedytile::math::convert::{f64_to_usize_saturating, usize_to_f64};
    use ndarray::Array3;

    // Tests exact ratio calculation for Red (2 pixels), Green (3 pixels), Blue (4 pixels) verifying correct denominator and exact ratios
    // Verified by testing ratio calculation uses correct denominator
    #[test]
    fn test_color_mapping_and_ratios() {
        let mut image_data = Array3::<f64>::zeros((3, 3, 4));

        // Red at 2 pixels, green at 3 and blue at 4, all opaque
        let pixels = [
            ((0, 0), 0),
            ((1, 1), 0),
            ((0, 1), 1),
            ((1, 0), 1),
            ((2, 0), 1),
            ((0, 2), 2),
            ((1, 2), 2),
            ((2, 1), 2),
            ((2, 2), 2),
        ];
        for ((row, col), channel) in pixels {
            for c in [channel, 3] {
                if let Some(val) = image_data.get_mut((row, col, c)) {
                    *val = 1.0;
                }
            }
        }

        let processor = ImageProcessor::from_raw_image(&image_data);

        assert_eq!(processor.unique_cell_count(), 3);

        let source_data = processor.source_data();
        assert_eq!(source_data.dim(), (3, 3));

        let mut color_counts = [0; 3];
        for &label in source_data {
            assert!((1..=3).contains(&label), "Invalid label: {label}");
            if let Some(count) = color_counts.get_mut(label - 1) {
                *count += 1;
            }
        }

        let total_pixels: usize = color_counts.iter().sum();
        assert_eq!(total_pixels, 9, "Total pixel count should be exactly 9");

        let ratios = processor.source_ratios();
        assert_eq!(ratios.len(), 3, "Should have exactly 3 color ratios");

        let sum: f64 = ratios.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Ratios must sum to exactly 1.0, got {sum}"
        );

        let expected_counts = [4, 3, 2];

        for (i, &expected_count) in expected_counts.iter().enumerate() {
            let expected_ratio = usize_to_f64(expected_count) / 9.0;
            let actual_ratio = ratios.get(i).copied().unwrap_or(0.0);

            assert!(
                (actual_ratio - expected_ratio).abs() < 1e-10,
                "Ratio {i} is incorrect. Expected {expected_ratio} (={expected_count}/9), got {actual_ratio}. This suggests denominator is wrong."
            );

            let wrong_ratio = usize_to_f64(expected_count) / 10.0;
            assert!(
                (actual_ratio - wrong_ratio).abs() > 0.01,
                "Ratio {i} seems to use wrong denominator (total+1)"
            );
        }

        let reconstructed_counts: Vec<usize> = ratios
            .iter()
            .map(|&r| f64_to_usize_saturating((r * 9.0).round()))
            .collect();

        assert_eq!(
            reconstructed_counts, expected_counts,
            "Reconstructed counts don't match expected counts"
        );

        assert!(
            ratios
                .first()
                .is_some_and(|&r| (r - 4.0 / 9.0).abs() < 1e-10),
            "Blue ratio must be exactly 4/9"
        );
        assert!(
            ratios
                .get(1)
                .is_some_and(|&r| (r - 3.0 / 9.0).abs() < 1e-10),
            "Green ratio must be exactly 3/9"
        );
        assert!(
            ratios
                .get(2)
                .is_some_and(|&r| (r - 2.0 / 9.0).abs() < 1e-10),
            "Red ratio must be exactly 2/9"
        );
    }

    // Tests edge cases with single color images verifying ratio is 1.0
    // Verified by testing ratio calculation denominator for edge cases
    #[test]
    fn test_ratio_calculation_edge_cases() {
        let mut image_data = Array3::<f64>::zeros((1, 1, 4));
        if let Some(val) = image_data.get_mut((0, 0, 0)) {
            *val = 1.0;
        }
        if let Some(val) = image_data.get_mut((0, 0, 3)) {
            *val = 1.0;
        }

        let processor = ImageProcessor::from_raw_image(&image_data);
        let ratios = processor.source_ratios();

        assert_eq!(ratios.len(), 1, "Should have exactly 1 color");
        assert!(
            ratios.first().is_some_and(|&r| (r - 1.0).abs() < 1e-10),
            "Single color should have ratio of 1.0"
        );

        let mut image_data_2 = Array3::<f64>::zeros((2, 1, 4));
        if let Some(val) = image_data_2.get_mut((0, 0, 1)) {
            *val = 1.0;
        }
        if let Some(val) = image_data_2.get_mut((0, 0, 3)) {
            *val = 1.0;
        }
        if let Some(val) = image_data_2.get_mut((1, 0, 1)) {
            *val = 1.0;
        }
        if let Some(val) = image_data_2.get_mut((1, 0, 3)) {
            *val = 1.0;
        }

        let processor_2 = ImageProcessor::from_raw_image(&image_data_2);
        let ratios_2 = processor_2.source_ratios();

        assert_eq!(ratios_2.len(), 1, "Should have exactly 1 color");
        assert!(
            ratios_2.first().is_some_and(|&r| (r - 1.0).abs() < 1e-10),
            "Single color should have ratio of 1.0"
        );

        assert!(
            ratios_2
                .first()
                .is_some_and(|&r| (r - 2.0 / 3.0).abs() > 0.3),
            "Ratio suggests wrong denominator calculation"
        );
    }
}

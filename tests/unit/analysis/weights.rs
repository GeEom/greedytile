//! Tests for top-k weight selection from probability matrices

#[cfg(test)]
mod tests {
    use crate::analysis::weights::top_k_valid_indices;
    use greedytile::algorithm::propagation::StepData;
    use greedytile::analysis::weights::calculate_position_selection;
    use greedytile::math::convert::usize_to_f64;
    use greedytile::spatial::GridState;
    use ndarray::Array2;
    use std::collections::BTreeMap;

    // Tests selection of k highest values from matrix
    // Verified by confirming that top_k_valid_indices selects the k highest values not lowest
    #[test]
    fn test_top_k_indices_selects_highest_values() {
        let mut matrix = Array2::<f64>::zeros((3, 3));
        let validity = Array2::<bool>::from_elem((3, 3), true);

        if let Some(val) = matrix.get_mut([0, 0]) {
            *val = 15.0;
        }
        if let Some(val) = matrix.get_mut([0, 1]) {
            *val = 10.0;
        }
        if let Some(val) = matrix.get_mut([0, 2]) {
            *val = 5.0;
        }
        if let Some(val) = matrix.get_mut([1, 0]) {
            *val = 12.0;
        }
        if let Some(val) = matrix.get_mut([1, 1]) {
            *val = 8.0;
        }
        if let Some(val) = matrix.get_mut([1, 2]) {
            *val = 3.0;
        }
        if let Some(val) = matrix.get_mut([2, 0]) {
            *val = 1.0;
        }
        if let Some(val) = matrix.get_mut([2, 1]) {
            *val = 6.0;
        }
        if let Some(val) = matrix.get_mut([2, 2]) {
            *val = 0.0;
        }

        let result = top_k_valid_indices(&matrix, &validity, 3);
        let mut values: Vec<f64> = result
            .iter()
            .filter_map(|&[i, j]| matrix.get([i, j]).copied())
            .collect();
        values.sort_by(|a, b| b.total_cmp(a));

        let expected = vec![15.0, 12.0, 10.0];
        assert_eq!(values, expected);

        let no_indices = top_k_valid_indices(&matrix, &validity, 0);
        assert_eq!(no_indices.len(), 0);
    }

    // Tests value ordering verification with various numeric distributions including mixed positive/negative values
    // Verified by proper value ordering and selection logic with comprehensive numeric scenarios
    #[test]
    fn test_top_k_indices_exact_count() {
        let mut matrix = Array2::<f64>::zeros((3, 3));
        let validity = Array2::<bool>::from_elem((3, 3), true);
        let values = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        for (idx, &val) in values.iter().enumerate() {
            let row = idx / 3;
            let col = idx % 3;
            if let Some(cell) = matrix.get_mut([row, col]) {
                *cell = val;
            }
        }

        let result = top_k_valid_indices(&matrix, &validity, 3);
        let mut actual_values: Vec<f64> = result
            .iter()
            .filter_map(|&[i, j]| matrix.get([i, j]).copied())
            .collect();

        actual_values.sort_by(|a, b| b.total_cmp(a));

        let expected = vec![9.0, 8.0, 7.0];
        assert_eq!(
            actual_values, expected,
            "Should return top 3 values sorted in descending order"
        );

        let mut mixed_matrix = Array2::<f64>::zeros((2, 2));
        let mixed_validity = Array2::<bool>::from_elem((2, 2), true);
        let mixed_values = [-5.0, 10.0, -1.0, 3.0];

        for (idx, &val) in mixed_values.iter().enumerate() {
            let row = idx / 2;
            let col = idx % 2;
            if let Some(cell) = mixed_matrix.get_mut([row, col]) {
                *cell = val;
            }
        }

        let mixed_result = top_k_valid_indices(&mixed_matrix, &mixed_validity, 2);
        let mut mixed_actual: Vec<f64> = mixed_result
            .iter()
            .filter_map(|&[i, j]| mixed_matrix.get([i, j]).copied())
            .collect();

        mixed_actual.sort_by(|a, b| b.total_cmp(a));

        assert_eq!(mixed_actual.len(), 2, "Should return exactly 2 values");
        let expected_mixed = vec![10.0, 3.0];
        assert_eq!(mixed_actual, expected_mixed, "Should return top 2 values");

        for window in mixed_actual.windows(2) {
            if let (Some(&curr), Some(&next)) = (window.first(), window.get(1)) {
                assert!(
                    curr >= next,
                    "Values should be in descending order: {curr} >= {next}"
                );
            }
        }
    }

    // Tests k=1, duplicate values, and proper selection
    // Verified by testing heap boundary condition for k selection
    #[test]
    fn test_top_k_indices_edge_cases() {
        let mut matrix = Array2::<f64>::zeros((2, 2));
        let validity = Array2::<bool>::from_elem((2, 2), true);
        if let Some(val) = matrix.get_mut([0, 0]) {
            *val = 1.0;
        }
        if let Some(val) = matrix.get_mut([0, 1]) {
            *val = 2.0;
        }
        if let Some(val) = matrix.get_mut([1, 0]) {
            *val = 3.0;
        }
        if let Some(val) = matrix.get_mut([1, 1]) {
            *val = 4.0;
        }

        let result = top_k_valid_indices(&matrix, &validity, 1);
        assert_eq!(result.len(), 1);
        if let Some(&[row, col]) = result.first() {
            let value = matrix.get([row, col]).copied().unwrap_or(0.0);
            assert!((value - 4.0).abs() < f64::EPSILON);
        }

        let mut matrix2 = Array2::<f64>::zeros((2, 3));
        let validity2 = Array2::<bool>::from_elem((2, 3), true);
        if let Some(val) = matrix2.get_mut([0, 0]) {
            *val = 5.0;
        }
        if let Some(val) = matrix2.get_mut([0, 1]) {
            *val = 5.0;
        }
        if let Some(val) = matrix2.get_mut([0, 2]) {
            *val = 3.0;
        }
        if let Some(val) = matrix2.get_mut([1, 0]) {
            *val = 3.0;
        }
        if let Some(val) = matrix2.get_mut([1, 1]) {
            *val = 1.0;
        }
        if let Some(val) = matrix2.get_mut([1, 2]) {
            *val = 1.0;
        }

        let result2 = top_k_valid_indices(&matrix2, &validity2, 4);
        assert_eq!(result2.len(), 4);

        let mut values2: Vec<f64> = result2
            .iter()
            .filter_map(|&[i, j]| matrix2.get([i, j]).copied())
            .collect();
        values2.sort_by(|a, b| b.total_cmp(a));

        let expected_top_4 = vec![5.0, 5.0, 3.0, 3.0];
        assert_eq!(values2, expected_top_4);
    }

    // Ensures no duplicate indices in results
    // Verified by confirming unique indices without duplicates
    #[test]
    fn test_top_k_indices_no_duplicates() {
        let mut matrix = Array2::<f64>::zeros((3, 3));
        let validity = Array2::<bool>::from_elem((3, 3), true);

        for i in 0..3 {
            for j in 0..3 {
                if let Some(val) = matrix.get_mut([i, j]) {
                    *val = usize_to_f64(i * 3 + j);
                }
            }
        }

        let result = top_k_valid_indices(&matrix, &validity, 3);
        assert_eq!(result.len(), 3);

        let mut unique_indices = std::collections::BTreeSet::new();
        for &[i, j] in &result {
            unique_indices.insert([i, j]);
        }
        assert_eq!(unique_indices.len(), 3);
    }

    // Tests invalid positions are excluded from selection
    // Verified by confirming positions marked as invalid are excluded from top-k selection
    #[test]
    fn test_top_k_indices_respects_validity() {
        let mut matrix = Array2::<f64>::zeros((3, 3));
        let mut validity = Array2::<bool>::from_elem((3, 3), true);

        for i in 0..3 {
            for j in 0..3 {
                if let Some(val) = matrix.get_mut([i, j]) {
                    *val = usize_to_f64(i * 3 + j);
                }
            }
        }

        if let Some(val) = validity.get_mut([2, 2]) {
            *val = false;
        }
        if let Some(val) = validity.get_mut([2, 1]) {
            *val = false;
        }
        if let Some(val) = validity.get_mut([2, 0]) {
            *val = false;
        }

        let result = top_k_valid_indices(&matrix, &validity, 3);
        let mut values: Vec<f64> = result
            .iter()
            .filter_map(|&[i, j]| matrix.get([i, j]).copied())
            .collect();
        values.sort_by(|a, b| b.total_cmp(a));

        let expected = vec![5.0, 4.0, 3.0];
        assert_eq!(values, expected);
    }

    // Tests that unnormalised stored probabilities cannot overflow the density bias
    // Verified by the bias exponentiating raw products to infinity before normalisation
    #[test]
    fn test_position_weights_finite_for_large_probabilities() {
        let mut grid_state = GridState::new(5, 5, 2);
        for probs in &mut grid_state.tile_probabilities {
            probs.fill(1e300);
        }
        grid_state.adjacency_weights.fill(3);

        let step_data = StepData {
            source_ratios: vec![0.5, 0.5],
            unique_cell_count: 2,
            grid_extension_radius: 5,
            density_correction_threshold: 0.1,
            density_correction_steepness: 0.05,
            density_minimum_strength: 0.1,
            source_tiles: Vec::new(),
            tile_compatibility_rules: BTreeMap::new(),
        };

        let result = calculate_position_selection(&grid_state, &[10, 0], &step_data, [0, 0]);

        assert!(
            result.weights.iter().all(|w| w.is_finite() && *w > 0.0),
            "Weights should be finite and positive, got {:?}",
            result.weights
        );
    }
}

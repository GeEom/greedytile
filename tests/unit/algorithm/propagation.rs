//! Tests for probability propagation and forced position pipeline

#[cfg(test)]
mod tests {

    use greedytile::algorithm::propagation::{
        ForcedPipeline, ForcedPosition, StepData, update_probabilities_and_entropy,
    };
    use greedytile::spatial::GridState;
    use ndarray::Array4;
    use std::collections::BTreeMap;

    // Tests duplicate forced positions are filtered by coordinates
    // Verified by removing the duplicate check in add_positions
    #[test]
    fn test_forced_pipeline_deduplication() -> crate::TestResult {
        let mut pipeline = ForcedPipeline::new();

        let positions = vec![
            ForcedPosition {
                coordinates: [1, 2],
                tile_reference: 5,
            },
            ForcedPosition {
                coordinates: [3, 4],
                tile_reference: 7,
            },
            ForcedPosition {
                coordinates: [5, 6],
                tile_reference: 9,
            },
        ];
        pipeline.add_positions(positions);

        let duplicates = vec![
            ForcedPosition {
                coordinates: [1, 2],
                tile_reference: 10,
            },
            ForcedPosition {
                coordinates: [7, 8],
                tile_reference: 11,
            },
            ForcedPosition {
                coordinates: [3, 4],
                tile_reference: 12,
            },
        ];
        pipeline.add_positions(duplicates);

        assert_eq!(
            pipeline.take_next().ok_or("pipeline ran out")?.coordinates,
            [1, 2]
        );
        assert_eq!(
            pipeline.take_next().ok_or("pipeline ran out")?.coordinates,
            [3, 4]
        );
        assert_eq!(
            pipeline.take_next().ok_or("pipeline ran out")?.coordinates,
            [5, 6]
        );
        assert_eq!(
            pipeline.take_next().ok_or("pipeline ran out")?.coordinates,
            [7, 8]
        );
        assert!(pipeline.take_next().is_none());
        Ok(())
    }

    // Tests basic probability update and entropy calculation
    // Verified by breaking the probability multiplication logic
    #[test]
    fn test_update_probabilities_and_entropy_basic() {
        let mut grid_state = GridState::new(3, 3, 2);

        for row in 0..3 {
            for col in 0..3 {
                for color in 0..2 {
                    if let Some(prob) = grid_state.tile_probabilities.get_mut(color)
                        && let Some(val) = prob.get_mut([row, col])
                    {
                        *val = 1.0;
                    }
                }
                if let Some(val) = grid_state.entropy.get_mut([row, col]) {
                    *val = 0.0;
                }
            }
        }

        let mut influence = Array4::<f64>::ones((2, 2, 3, 3));
        if let Some(val) = influence.get_mut([0, 0, 1, 1]) {
            *val = 0.5;
        }
        if let Some(val) = influence.get_mut([0, 1, 1, 1]) {
            *val = 2.0;
        }

        let step_data = StepData {
            source_ratios: vec![0.5, 0.5],
            unique_cell_count: 2,
            grid_extension_radius: 1,
            density_correction_threshold: 0.1,
            density_correction_steepness: 0.05,
            density_minimum_strength: 0.1,
            source_tiles: vec![],
            tile_compatibility_rules: BTreeMap::new(),
        };

        update_probabilities_and_entropy(
            &mut grid_state,
            &influence,
            1,
            [0, 0],
            [1, 1],
            &step_data,
        );

        assert!(
            grid_state
                .tile_probabilities
                .first()
                .and_then(|probs| probs.get([1, 1]))
                .is_some_and(|&v| (v - 0.5).abs() < 1e-10)
        );
        assert!(
            grid_state
                .tile_probabilities
                .get(1)
                .and_then(|probs| probs.get([1, 1]))
                .is_some_and(|&v| (v - 2.0).abs() < 1e-10)
        );

        let expected_entropy = 0.4_f64.mul_add((0.4_f64).ln(), 1.6 * (1.6_f64).ln());
        assert!(
            grid_state
                .entropy
                .get([1, 1])
                .is_some_and(|&v| (v - expected_entropy).abs() < 1e-10)
        );
    }

    // Tests asymmetric influence patterns verify correct indexing
    // Verified by transposing influence matrix indices
    #[test]
    fn test_update_probabilities_and_entropy_complex() {
        let mut grid_state = GridState::new(5, 5, 3);
        for (probs, value) in grid_state
            .tile_probabilities
            .iter_mut()
            .zip([0.8, 1.2, 0.6])
        {
            probs.fill(value);
        }
        grid_state.entropy.fill(0.0);

        // Each corner of the 5x5 influence window has its own factor
        let corner_factor = |cell: [usize; 2]| match cell {
            [0, 0] => 10.0,
            [0, 4] => 7.0,
            [4, 0] => 3.0,
            [4, 4] => 2.0,
            _ => 1.0,
        };
        let influence = Array4::from_shape_fn((3, 3, 5, 5), |(_, _, i, j)| corner_factor([i, j]));

        let step_data = StepData {
            source_ratios: vec![0.33, 0.33, 0.34],
            unique_cell_count: 3,
            grid_extension_radius: 2,
            density_correction_threshold: 0.1,
            density_correction_steepness: 0.05,
            density_minimum_strength: 0.1,
            source_tiles: vec![],
            tile_compatibility_rules: BTreeMap::new(),
        };

        update_probabilities_and_entropy(
            &mut grid_state,
            &influence,
            2,
            [0, 0],
            [2, 2],
            &step_data,
        );

        for cell in [[0, 0], [0, 4], [4, 0], [4, 4], [1, 1]] {
            let expected = 0.8 * corner_factor(cell);
            assert!(
                grid_state
                    .tile_probabilities
                    .first()
                    .and_then(|probs| probs.get(cell))
                    .is_some_and(|&v| (v - expected).abs() < 1e-10),
                "Position {cell:?} should be scaled by {}",
                corner_factor(cell)
            );
        }
    }

    // Tests edge case where mean probability approaches zero
    // Verified by returning non-zero entropy when mean is near zero
    #[test]
    fn test_update_probabilities_and_entropy_edge_case_zero_mean() {
        let mut grid_state = GridState::new(3, 3, 2);

        for row in 0..3 {
            for col in 0..3 {
                if let Some(prob) = grid_state.tile_probabilities.get_mut(0)
                    && let Some(val) = prob.get_mut([row, col])
                {
                    *val = 1e-20;
                }
                if let Some(prob) = grid_state.tile_probabilities.get_mut(1)
                    && let Some(val) = prob.get_mut([row, col])
                {
                    *val = 1e-20;
                }
                if let Some(val) = grid_state.entropy.get_mut([row, col]) {
                    *val = 0.0;
                }
            }
        }

        let mut influence = Array4::<f64>::zeros((2, 2, 3, 3));
        for i in 0..2 {
            for j in 0..2 {
                for row in 0..3 {
                    for col in 0..3 {
                        if let Some(val) = influence.get_mut([i, j, row, col]) {
                            *val = 1e-20;
                        }
                    }
                }
            }
        }

        let step_data = StepData {
            source_ratios: vec![0.5, 0.5],
            unique_cell_count: 2,
            grid_extension_radius: 1,
            density_correction_threshold: 0.1,
            density_correction_steepness: 0.05,
            density_minimum_strength: 0.1,
            source_tiles: vec![],
            tile_compatibility_rules: BTreeMap::new(),
        };

        update_probabilities_and_entropy(
            &mut grid_state,
            &influence,
            1,
            [0, 0],
            [1, 1],
            &step_data,
        );

        assert!(
            grid_state
                .entropy
                .get([1, 1])
                .is_some_and(|&v| v.abs() < f64::EPSILON)
        );
    }
}

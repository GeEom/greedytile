//! Tests that deadlock resolution unlocks tiles and rewinds what they changed

#[cfg(test)]
mod tests {
    use greedytile::algorithm::deadlock::{DeadlockState, resolve_spatial_deadlock};
    use greedytile::algorithm::feasibility::FeasibilityCountLayer;
    use greedytile::algorithm::propagation::StepData;
    use greedytile::io::configuration::ADJACENCY_LEVELS;
    use greedytile::spatial::GridState;
    use ndarray::Array4;
    use std::collections::BTreeMap;

    /// A 2x2 block of placed tiles: locked value 2 is tile type 1, 3 is type 2
    const PLACED: [([usize; 2], u32); 4] = [([1, 1], 2), ([1, 2], 3), ([2, 1], 2), ([2, 2], 3)];

    /// The probability factor tile type `tile` applies to colour `color`
    const fn influence(tile: usize, color: usize) -> f64 {
        if tile == color { 1.5 } else { 0.5 }
    }

    /// A 5x5 grid holding `PLACED`, with the weights and probabilities placing it left
    fn placed_grid() -> Result<GridState, String> {
        let mut grid_state = GridState::new(5, 5, 2);

        for ([row, col], locked) in PLACED {
            *grid_state
                .locked_tiles
                .get_mut([row, col])
                .ok_or("cell outside grid")? = locked;
            let tile = usize::try_from(locked - 2).map_err(|e| e.to_string())?;
            for color in 0..2 {
                let probs = grid_state
                    .tile_probabilities
                    .get_mut(color)
                    .ok_or("no colour")?;
                *probs.get_mut([row, col]).ok_or("cell outside grid")? *= influence(tile, color);
            }
        }

        for row in 0..5_usize {
            for col in 0..5_usize {
                let mut weight = 1;
                for ([locked_row, locked_col], _) in PLACED {
                    let dist = row.abs_diff(locked_row).max(col.abs_diff(locked_col));
                    if (1..=ADJACENCY_LEVELS).contains(&dist) {
                        weight += u32::try_from(1 + ADJACENCY_LEVELS - dist)
                            .map_err(|e| e.to_string())?;
                    }
                }
                *grid_state
                    .adjacency_weights
                    .get_mut([row, col])
                    .ok_or("cell outside grid")? = weight;
            }
        }

        Ok(grid_state)
    }

    fn probability(grid_state: &GridState, color: usize, cell: [usize; 2]) -> Result<f64, String> {
        grid_state
            .tile_probabilities
            .get(color)
            .and_then(|probs| probs.get(cell))
            .copied()
            .ok_or_else(|| format!("no probability for colour {color} at {cell:?}"))
    }

    // Verified by removing the unlocking step from resolve_spatial_deadlock
    #[test]
    fn test_deadlock_resolution_unlocks_tiles_and_reverts_state() -> Result<(), String> {
        let mut grid_state = placed_grid()?;
        let mut feasibility_layer = FeasibilityCountLayer::new(5, 5, 2);
        let mut selection_tally = vec![2, 2];
        let step_data = StepData {
            source_ratios: vec![0.5, 0.5],
            unique_cell_count: 2,
            grid_extension_radius: 2,
            density_correction_threshold: 0.5,
            density_correction_steepness: 10.0,
            density_minimum_strength: 0.1,
            source_tiles: vec![
                [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[2, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            tile_compatibility_rules: BTreeMap::new(),
        };
        let influence_matrices =
            Array4::from_shape_fn((2, 2, 5, 5), |(tile, color, _, _)| influence(tile, color));

        let result = resolve_spatial_deadlock(
            DeadlockState {
                grid_state: &mut grid_state,
                feasibility_layer: &mut feasibility_layer,
                selection_tally: &mut selection_tally,
                visualization: &mut None,
            },
            [2, 2],
            [0, 0],
            &step_data,
            &influence_matrices,
            0,
        );

        // The first deadlock here has radius 1, which reaches the whole block
        assert_eq!(result.tiles_unlocked, 4);
        assert_eq!(result.unlocked_positions.len(), 4);
        for ([row, col], _) in PLACED {
            assert_eq!(grid_state.locked_tiles.get([row, col]).copied(), Some(1));
        }
        assert_eq!(selection_tally, vec![0, 0]);
        assert_eq!(grid_state.adjacency_weights.get([2, 2]).copied(), Some(0));

        // Each cell divides out all four removals, two at 1.5 and two at 0.5
        let raised = 1.5 / (1.5 * 1.5 * 0.5 * 0.5);
        let lowered = 0.5 / (1.5 * 1.5 * 0.5 * 0.5);
        assert!((probability(&grid_state, 0, [1, 1])? - raised).abs() < 1e-9);
        assert!((probability(&grid_state, 1, [1, 1])? - lowered).abs() < 1e-9);
        assert!((probability(&grid_state, 0, [2, 2])? - lowered).abs() < 1e-9);
        assert!((probability(&grid_state, 1, [2, 2])? - raised).abs() < 1e-9);
        Ok(())
    }
}

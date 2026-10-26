//! Tests for tile selection algorithms and density correction

#[cfg(test)]
mod tests {

    use greedytile::algorithm::cache::ViableTilesCache;
    use greedytile::algorithm::propagation::StepData;
    use greedytile::algorithm::selection::{
        compute_viable_tiles_at_position, get_tile_probabilities_at_position,
        optimal_density_correction,
    };
    use greedytile::spatial::GridState;
    use greedytile::spatial::tiles::Tile;
    use std::collections::BTreeMap;

    // Tests viable tile computation with constraints
    // Verified by testing wildcard values (-1) in patterns match any tile value
    #[test]
    fn test_compute_viable_tiles_at_position_basic() {
        let mut grid_state = GridState::new(5, 5, 2);

        let source_tiles: Vec<Tile> = vec![
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        ];

        if let Some(val) = grid_state.locked_tiles.get_mut([1, 1]) {
            *val = 1;
        }
        if let Some(val) = grid_state.locked_tiles.get_mut([1, 2]) {
            *val = 2;
        }

        let mut dispatch_rules = BTreeMap::new();
        dispatch_rules.insert(vec![1, 0], vec![1]);
        dispatch_rules.insert(vec![0, 1], vec![2]);
        dispatch_rules.insert(vec![1, 1], vec![1, 2]);
        dispatch_rules.insert(vec![0, 0], vec![1, 2]);

        let step_data = StepData {
            source_ratios: vec![0.5, 0.5],
            unique_cell_count: 2,
            grid_extension_radius: 5,
            density_correction_threshold: 0.1,
            density_correction_steepness: 0.05,
            density_minimum_strength: 0.1,
            source_tiles: source_tiles.clone(),
            tile_compatibility_rules: dispatch_rules,
        };

        let mut cache = ViableTilesCache::new();
        let viable = compute_viable_tiles_at_position(
            &grid_state,
            [2, 2],
            [0, 0],
            &source_tiles,
            &step_data,
            &mut cache,
        );

        assert!(!viable.is_empty(), "Should have at least one viable tile");
        assert!(
            viable.len() <= 2,
            "Should not exceed number of unique values"
        );
    }

    // Tests all tiles viable with wildcards
    // Verified by confirming -1 is the wildcard value that matches any tile
    #[test]
    fn test_compute_viable_tiles_with_only_wildcards() {
        let grid_state = GridState::new(5, 5, 2);

        let source_tiles: Vec<Tile> = vec![
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
        ];

        let mut dispatch_rules = BTreeMap::new();
        dispatch_rules.insert(vec![0, 0], vec![1, 2]);

        let step_data = StepData {
            source_ratios: vec![0.5, 0.5],
            unique_cell_count: 2,
            grid_extension_radius: 5,
            density_correction_threshold: 0.1,
            density_correction_steepness: 0.05,
            density_minimum_strength: 0.1,
            source_tiles: source_tiles.clone(),
            tile_compatibility_rules: dispatch_rules,
        };

        let mut cache = ViableTilesCache::new();
        let viable = compute_viable_tiles_at_position(
            &grid_state,
            [2, 2],
            [0, 0],
            &source_tiles,
            &step_data,
            &mut cache,
        );

        assert_eq!(
            viable.len(),
            2,
            "All tiles should be viable with only wildcards"
        );
        assert!(viable.contains(&1), "Should contain value 1");
        assert!(viable.contains(&2), "Should contain value 2");
    }

    // Tests heavily constrained position
    // Verified by testing locked tile value conversion from 1-indexed to 0-indexed
    #[test]
    fn test_compute_viable_tiles_constrained_position() {
        let mut grid_state = GridState::new(5, 5, 2);

        let source_tiles: Vec<Tile> = vec![
            [[1, 2, 1], [2, 1, 2], [1, 2, 1]],
            [[2, 1, 2], [1, 2, 1], [2, 1, 2]],
        ];

        if let Some(val) = grid_state.locked_tiles.get_mut([1, 1]) {
            *val = 1;
        }
        if let Some(val) = grid_state.locked_tiles.get_mut([1, 2]) {
            *val = 2;
        }
        if let Some(val) = grid_state.locked_tiles.get_mut([2, 1]) {
            *val = 2;
        }
        if let Some(val) = grid_state.locked_tiles.get_mut([2, 2]) {
            *val = 1;
        }

        let mut dispatch_rules = BTreeMap::new();
        dispatch_rules.insert(vec![1, 1], vec![1, 2]);
        dispatch_rules.insert(vec![0, 0], vec![1, 2]);
        dispatch_rules.insert(vec![1, 0], vec![1, 2]);
        dispatch_rules.insert(vec![0, 1], vec![1, 2]);

        let step_data = StepData {
            source_ratios: vec![0.5, 0.5],
            unique_cell_count: 2,
            grid_extension_radius: 5,
            density_correction_threshold: 0.1,
            density_correction_steepness: 0.05,
            density_minimum_strength: 0.1,
            source_tiles: source_tiles.clone(),
            tile_compatibility_rules: dispatch_rules,
        };

        let mut cache = ViableTilesCache::new();
        let viable = compute_viable_tiles_at_position(
            &grid_state,
            [1, 1],
            [0, 0],
            &source_tiles,
            &step_data,
            &mut cache,
        );

        assert!(
            !viable.is_empty(),
            "Should have viable values for checkerboard position"
        );
    }

    // Tests density correction favors underrepresented tiles
    // Verified by testing correction sign for density balancing
    #[test]
    fn test_optimal_density_correction_balances_underrepresented_tiles() {
        let source_ratios = vec![0.5, 0.3, 0.2];
        let present_tally = vec![40, 35, 25];
        let total_placed = 100;

        let deviations = vec![0.4 - 0.5, 0.35 - 0.3, 0.25 - 0.2];

        let probabilities = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

        let corrections = optimal_density_correction(
            &probabilities,
            &present_tally,
            &source_ratios,
            total_placed,
            &deviations,
        );

        assert_eq!(corrections.len(), 3);

        assert!(
            corrections.first().copied().unwrap_or(0.0) > 0.0,
            "Underrepresented tile should receive positive correction, got: {}",
            corrections.first().copied().unwrap_or(0.0)
        );

        assert!(
            corrections.get(1).copied().unwrap_or(0.0) < 0.0,
            "Overrepresented tile should receive negative correction, got: {}",
            corrections.get(1).copied().unwrap_or(0.0)
        );
        assert!(
            corrections.get(2).copied().unwrap_or(0.0) < 0.0,
            "Overrepresented tile should receive negative correction, got: {}",
            corrections.get(2).copied().unwrap_or(0.0)
        );

        assert!(
            corrections.first().copied().unwrap_or(0.0).abs()
                > corrections.get(1).copied().unwrap_or(0.0).abs(),
            "Larger deviation should produce larger correction magnitude"
        );
    }

    // Tests that stored products far outside [0, 1] come back as a distribution
    // Verified by the density correction blowing up on raw values before normalisation
    #[test]
    fn test_tile_probabilities_are_normalised() {
        let mut grid_state = GridState::new(3, 3, 3);
        for (color, value) in [1e300, 1e-300, 3e299].into_iter().enumerate() {
            if let Some(prob) = grid_state
                .tile_probabilities
                .get_mut(color)
                .and_then(|probs| probs.get_mut([1, 1]))
            {
                *prob = value;
            }
        }

        let probabilities = get_tile_probabilities_at_position(&grid_state, [1, 1], [0, 0]);

        let total: f64 = probabilities.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "Should sum to one, got {total}"
        );
        let first = probabilities.first().copied().unwrap_or(0.0);
        let third = probabilities.get(2).copied().unwrap_or(0.0);
        assert!(
            (first / third - 1e300 / 3e299).abs() < 1e-9,
            "Should keep the stored ratios, got {probabilities:?}"
        );
    }

    // Tests the fallback when every stored product has underflowed to zero
    // Verified by confirming no NaN reaches the tile draw
    #[test]
    fn test_tile_probabilities_underflow_is_uniform() {
        let mut grid_state = GridState::new(3, 3, 4);
        for probs in &mut grid_state.tile_probabilities {
            probs.fill(0.0);
        }

        let probabilities = get_tile_probabilities_at_position(&grid_state, [1, 1], [0, 0]);

        assert_eq!(probabilities, vec![0.25; 4]);
    }
}

//! Tests for dynamic grid state management and matrix extension

#[cfg(test)]
mod tests {

    use crate::spatial::grid::extend_matrices;
    use ndarray::Array3;

    // Tests grid expansion for out-of-bounds coordinates with content preservation
    // Verified by omitting data copy loop in extend_array_3d
    #[test]
    fn test_extend_matrices_expands_grid_for_out_of_bounds_coordinates() {
        let mut initial = Array3::<f64>::zeros((2, 3, 3));
        if let Some(val) = initial.get_mut([0, 1, 1]) {
            *val = 5.0;
        }
        if let Some(val) = initial.get_mut([1, 1, 1]) {
            *val = 10.0;
        }

        let offset = [1, 1];

        let coordinates = [5, 5];
        let radius = 2;

        let (extended, new_offset) = extend_matrices(initial.clone(), offset, &coordinates, radius);

        assert_ne!(
            extended.dim(),
            initial.dim(),
            "Grid should have been extended"
        );

        assert!(extended.dim().1 >= 9, "Grid height should be at least 9");
        assert!(extended.dim().2 >= 9, "Grid width should be at least 9");

        let padding_left = (new_offset[0] - offset[0]) as usize;
        let padding_top = (new_offset[1] - offset[1]) as usize;

        assert!(
            extended
                .get([0, 1 + padding_left, 1 + padding_top])
                .is_some_and(|&v| (v - 5.0).abs() < f64::EPSILON)
        );
        assert!(
            extended
                .get([1, 1 + padding_left, 1 + padding_top])
                .is_some_and(|&v| (v - 10.0).abs() < f64::EPSILON)
        );

        let last_row = extended.dim().1 - 1;
        let last_col = extended.dim().2 - 1;
        assert!(
            extended
                .get([0, last_row, last_col])
                .is_some_and(|&v| (v - 1.0).abs() < f64::EPSILON)
        );
        assert!(
            extended
                .get([1, last_row, last_col])
                .is_some_and(|&v| (v - 1.0).abs() < f64::EPSILON)
        );
    }

    // Tests GridState extension preserves all data (entropy, locked tiles, adjacency weights, probabilities, feasibility)
    // Verified by skipping copy at position [1,1]
    #[test]
    fn test_gridstate_extend_if_needed_preserves_existing_data() {
        use crate::spatial::grid::GridState;

        let mut grid = GridState::new(3, 3, 2);
        if let Some(val) = grid.entropy.get_mut([1, 1]) {
            *val = 0.5;
        }
        if let Some(val) = grid.locked_tiles.get_mut([0, 2]) {
            *val = 5;
        }
        if let Some(val) = grid.adjacency_weights.get_mut([2, 0]) {
            *val = 3;
        }
        if let Some(prob) = grid.tile_probabilities.get_mut(0)
            && let Some(val) = prob.get_mut([1, 2]) {
                *val = 0.7;
            }
        if let Some(prob) = grid.tile_probabilities.get_mut(1)
            && let Some(val) = prob.get_mut([2, 1]) {
                *val = 0.3;
            }
        if let Some(val) = grid.feasibility.get_mut([0, 0]) {
            *val = 0.9;
        }

        let _original_dims = (grid.rows(), grid.cols());

        let offset = [0, 0];
        let coordinates = [5, 5];
        let radius = 2;

        let (new_offset, extended) = grid.extend_if_needed(offset, &coordinates, radius);

        assert!(extended, "Grid should have been extended");
        assert_eq!(
            new_offset,
            [0, 0],
            "Offset should remain the same (no left/top padding needed)"
        );

        assert_eq!(grid.rows(), 8, "Grid should have 8 rows");
        assert_eq!(grid.cols(), 8, "Grid should have 8 columns");

        assert!(
            grid.entropy
                .get([1, 1])
                .is_some_and(|&v| (v - 0.5).abs() < f64::EPSILON),
            "Entropy value should be preserved"
        );
        assert_eq!(
            grid.locked_tiles.get([0, 2]).copied(),
            Some(5),
            "Locked tile should be preserved"
        );
        assert_eq!(
            grid.adjacency_weights.get([2, 0]).copied(),
            Some(3),
            "Adjacency weight should be preserved"
        );
        assert!(
            grid.tile_probabilities
                .first()
                .and_then(|probs| probs.get([1, 2]))
                .is_some_and(|&v| (v - 0.7).abs() < f64::EPSILON),
            "Tile probability 0 should be preserved"
        );
        assert!(
            grid.tile_probabilities
                .get(1)
                .and_then(|probs| probs.get([2, 1]))
                .is_some_and(|&v| (v - 0.3).abs() < f64::EPSILON),
            "Tile probability 1 should be preserved"
        );
        assert!(
            grid.feasibility
                .get([0, 0])
                .is_some_and(|&v| (v - 0.9).abs() < f64::EPSILON),
            "Feasibility should be preserved"
        );

        assert!(
            grid.entropy
                .get([0, 0])
                .is_some_and(|&v| (v - 1.0).abs() < f64::EPSILON),
            "New cells should have entropy 1.0"
        );
        assert_eq!(
            grid.locked_tiles.get([7, 7]).copied(),
            Some(1),
            "New cells should have locked_tiles 1"
        );
        assert_eq!(
            grid.adjacency_weights.get([0, 7]).copied(),
            Some(1),
            "New cells should have adjacency_weights 1"
        );
        assert!(
            grid.tile_probabilities
                .first()
                .and_then(|probs| probs.get([7, 0]))
                .is_some_and(|&v| (v - 1.0).abs() < f64::EPSILON),
            "New cells should have probability 1.0"
        );
        assert!(
            grid.feasibility
                .get([7, 7])
                .is_some_and(|&v| (v - 1.0).abs() < f64::EPSILON),
            "New cells should have feasibility 1.0"
        );
    }
}

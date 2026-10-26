//! Tests for feasibility tracking layer with tile dispatch and counting

#[cfg(test)]
mod tests {
    use greedytile::algorithm::feasibility::FeasibilityCountLayer;
    use greedytile::spatial::tiles::Tile;
    use std::collections::BTreeMap;

    // Tests new layer has fraction 1.0 everywhere
    // Verified by initializing counts with 0 instead of tile_count
    #[test]
    fn test_feasibility_count_layer_new() {
        let layer = FeasibilityCountLayer::new(5, 5, 10);

        assert!((layer.get_fraction(0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((layer.get_fraction(4, 4) - 1.0).abs() < f64::EPSILON);
    }

    // Verifies extending grid preserves existing data
    // Verified by removing data copying logic during extension
    #[test]
    fn test_feasibility_count_layer_extend_preserves_data() {
        let mut layer = FeasibilityCountLayer::new(2, 2, 10);

        let source_tiles: Vec<Tile> = vec![
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
        ];
        let mut dispatch_rules = BTreeMap::new();
        dispatch_rules.insert(vec![0; 10], vec![1, 2]);

        let tile_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
        layer.update_count(0, 0, &tile_grid, &source_tiles, &dispatch_rules, 10);

        let original_fraction = layer.get_fraction(0, 0);

        layer.extend_to(4, 3);

        assert!((layer.get_fraction(0, 0) - original_fraction).abs() < f64::EPSILON);
        let fraction_1_1 = layer.get_fraction(1, 1);
        assert!((layer.get_fraction(1, 1) - fraction_1_1).abs() < f64::EPSILON);

        assert!((layer.get_fraction(2, 0) - 1.0).abs() < f64::EPSILON);
        assert!((layer.get_fraction(3, 2) - 1.0).abs() < f64::EPSILON);

        layer.extend_to(2, 2);

        assert!((layer.get_fraction(0, 0) - original_fraction).abs() < f64::EPSILON);
    }

    // Tests update count with all matching tiles
    // Verified by commenting out the count increment
    #[test]
    fn test_update_count_with_matching_tiles() {
        let mut layer = FeasibilityCountLayer::new(3, 3, 3);

        let source_tiles: Vec<Tile> = vec![
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[2, 3, 4], [5, 6, 7], [8, 9, 1]],
            [[3, 4, 5], [6, 7, 8], [9, 1, 2]],
        ];

        let mut dispatch_rules = BTreeMap::new();
        dispatch_rules.insert(vec![0; 10], vec![1, 2, 3]);

        let tile_grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];

        layer.update_count(0, 0, &tile_grid, &source_tiles, &dispatch_rules, 10);

        assert!((layer.get_fraction(0, 0) - 1.0).abs() < f64::EPSILON);
    }

    // Tests partial matches produce correct fraction
    // Verified by doubling the count increment
    #[test]
    fn test_update_count_with_partial_matches() {
        let mut layer = FeasibilityCountLayer::new(3, 3, 3);

        let source_tiles: Vec<Tile> = vec![
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 2, 0], [4, 5, 0], [7, 8, 0]],
            [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
        ];

        let mut dispatch_rules = BTreeMap::new();
        dispatch_rules.insert(vec![1, 1, 0, 1, 1, 0, 1, 1, 0, 0], vec![1, 2, 3]);

        let tile_grid = [[1, 2, 0], [4, 5, 0], [7, 8, 0]];

        layer.update_count(1, 1, &tile_grid, &source_tiles, &dispatch_rules, 10);

        assert!((layer.get_fraction(1, 1) - 2.0 / 3.0).abs() < f64::EPSILON);
    }

    // Verifies out of bounds returns 1.0
    // Verified by changing return value to 0.0
    #[test]
    fn test_get_fraction_out_of_bounds() {
        let layer = FeasibilityCountLayer::new(3, 3, 10);

        assert!((layer.get_fraction(5, 5) - 1.0).abs() < f64::EPSILON);
        assert!((layer.get_fraction(100, 100) - 1.0).abs() < f64::EPSILON);
    }

    // Tests no-op extension preserves data
    // Verified by resetting values instead of preserving
    #[test]
    fn test_extend_to_same_dimensions() {
        let mut layer = FeasibilityCountLayer::new(3, 3, 10);

        let source_tiles: Vec<Tile> = vec![[[1; 3]; 3]];
        let dispatch_rules = BTreeMap::new();
        let tile_grid = [[1; 3]; 3];

        layer.update_count(1, 1, &tile_grid, &source_tiles, &dispatch_rules, 10);
        let fraction = layer.get_fraction(1, 1);

        layer.extend_to(3, 3);

        assert!((layer.get_fraction(1, 1) - fraction).abs() < f64::EPSILON);
    }
}

//! Tests for tile extraction with rotation and reflection transformations

#[cfg(test)]
mod tests {

    use greedytile::spatial::tiles::{Tile, TileExtractor};
    use ndarray::Array2;

    fn rotate_90_reference(tile: &Tile) -> Tile {
        let n = 3;
        let mut rotated = [[0; 3]; 3];
        for i in 0..n {
            for j in 0..n {
                if let Some(row) = tile.get(n - 1 - j)
                    && let Some(&val) = row.get(i)
                        && let Some(rot_row) = rotated.get_mut(i)
                            && let Some(rot_val) = rot_row.get_mut(j) {
                                *rot_val = val;
                            }
            }
        }
        rotated
    }

    fn reflect_reference(tile: &Tile) -> Tile {
        let n = 3;
        let mut reflected = [[0; 3]; 3];
        for i in 0..n {
            for j in 0..n {
                if let Some(row) = tile.get(i)
                    && let Some(&val) = row.get(n - 1 - j)
                        && let Some(ref_row) = reflected.get_mut(i)
                            && let Some(ref_val) = ref_row.get_mut(j) {
                                *ref_val = val;
                            }
            }
        }
        reflected
    }

    // Tests tile extraction with rotations and reflections producing expected counts
    // Verified by disabling rotation addition
    #[test]
    fn test_tile_transformations() {
        let source_data = Array2::from_shape_vec(
            (5, 5),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25,
            ],
        )
        .unwrap();

        let extractor_base = TileExtractor::extract_tiles(&source_data, 3, false, false);
        let base_tiles = extractor_base.source_tiles();

        let extractor_rot = TileExtractor::extract_tiles(&source_data, 3, true, false);
        let rot_tiles = extractor_rot.source_tiles();

        let extractor_all = TileExtractor::extract_tiles(&source_data, 3, true, true);
        let all_tiles = extractor_all.source_tiles();

        assert_eq!(base_tiles.len(), 9, "Should extract 9 unique base tiles");

        assert!(
            rot_tiles.len() > base_tiles.len(),
            "Rotations should create more unique tiles"
        );
        assert!(
            rot_tiles.len() <= base_tiles.len() * 4,
            "Rotations can at most quadruple the tile count"
        );

        assert!(
            all_tiles.len() > rot_tiles.len(),
            "Adding reflections should create more unique tiles"
        );
        assert!(
            all_tiles.len() <= base_tiles.len() * 8,
            "Rotations and reflections can at most create 8x tiles"
        );

        let expected_first_tile = [[1, 2, 3], [6, 7, 8], [11, 12, 13]];
        assert_eq!(
            base_tiles.first().copied(),
            Some(expected_first_tile),
            "First tile should match expected pattern"
        );

        for tile in all_tiles {
            assert_eq!(tile.len(), 3);
            for row in tile {
                assert_eq!(row.len(), 3);
            }

            for row in tile {
                for &value in row {
                    assert!(
                        (1..=25).contains(&value),
                        "Tile values should be from source data"
                    );
                }
            }
        }
    }

    // Tests 90-degree rotation produces correct mapping
    // Verified by returning unchanged tile
    #[test]
    fn test_rotate_90_correctness() {
        let test_tile: Tile = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];

        let expected_rot90: Tile = [[7, 4, 1], [8, 5, 2], [9, 6, 3]];

        let source_data = Array2::from_shape_vec((3, 3), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();

        let extractor = TileExtractor::extract_tiles(&source_data, 3, true, false);
        let tiles = extractor.source_tiles();

        let rot90_found = tiles.contains(&expected_rot90);
        assert!(
            rot90_found,
            "90-degree rotation not found in extracted tiles. Expected: {expected_rot90:?}"
        );

        let rot90 = rotate_90_reference(&test_tile);
        let rot180 = rotate_90_reference(&rot90);
        let rot270 = rotate_90_reference(&rot180);
        let rot360 = rotate_90_reference(&rot270);

        assert_eq!(rot90, expected_rot90, "90-degree rotation is incorrect");
        assert_eq!(
            rot180,
            [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
            "180-degree rotation is incorrect"
        );
        assert_eq!(
            rot270,
            [[3, 6, 9], [2, 5, 8], [1, 4, 7]],
            "270-degree rotation is incorrect"
        );
        assert_eq!(
            rot360, test_tile,
            "360-degree rotation should return to original"
        );

        assert!(
            tiles.contains(&test_tile),
            "Original tile should be present"
        );
        assert!(tiles.contains(&rot90), "90° rotation should be present");
        assert!(tiles.contains(&rot180), "180° rotation should be present");
        assert!(tiles.contains(&rot270), "270° rotation should be present");
    }

    // Tests detection of rotation formula bug (using j,i instead of n-1-j,i)
    // Verified by using incorrect formula
    #[test]
    fn test_rotate_90_bug_detection() {
        let expected_correct: Tile = [[0, 0, 1], [0, 0, 0], [0, 0, 0]];

        let buggy_result: Tile = [[1, 0, 0], [0, 0, 0], [0, 0, 0]];

        assert_ne!(
            expected_correct, buggy_result,
            "Bug detection pattern: correct and buggy results should differ"
        );

        let source_data = Array2::from_shape_vec((3, 3), vec![1, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        let extractor = TileExtractor::extract_tiles(&source_data, 3, true, false);
        let tiles = extractor.source_tiles();

        assert!(
            tiles.contains(&expected_correct),
            "Correct 90-degree rotation not found. This indicates the rotation bug is present!"
        );

        assert!(
            tiles.iter().filter(|&&tile| tile == buggy_result).count() <= 1,
            "Found buggy rotation result multiple times. The rotation formula is incorrect!"
        );
    }

    // Tests horizontal reflection produces correct mapping
    // Verified by returning unchanged tile
    #[test]
    fn test_reflection_correctness() {
        let test_tile: Tile = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];

        let expected_reflected: Tile = [[3, 2, 1], [6, 5, 4], [9, 8, 7]];

        let reflected = reflect_reference(&test_tile);
        assert_eq!(
            reflected, expected_reflected,
            "Horizontal reflection is incorrect"
        );

        let double_reflected = reflect_reference(&reflected);
        assert_eq!(
            double_reflected, test_tile,
            "Double reflection should return to original"
        );

        let source_data = Array2::from_shape_vec((3, 3), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let extractor = TileExtractor::extract_tiles(&source_data, 3, false, true);
        let tiles = extractor.source_tiles();

        assert!(
            tiles.contains(&expected_reflected),
            "Reflected tile not found in extracted tiles"
        );
    }
}

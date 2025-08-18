//! Tile extraction and pattern matching utilities
//!
//! Extracts overlapping tiles from source images and builds boolean reference
//! rules for constraint-based pattern matching. Supports transformations
//! (rotation, reflection) to increase pattern variety from limited source data.

use ndarray::Array2;
use std::collections::{HashMap, HashSet};

/// A 3x3 tile with cell values representing color/type indices
pub type Tile = [[usize; 3]; 3];

/// Tile extractor managing source patterns and constraint rules
///
/// Maintains deduplicated tiles and boolean membership rules for efficient
/// pattern matching during wave function collapse.
pub struct TileExtractor {
    source_tiles: Vec<Tile>,
    source_tile_boolean_reference_rules: HashMap<Vec<u8>, Vec<usize>>,
}

impl TileExtractor {
    /// Extract tiles from source data with optional transformations
    ///
    /// Slides a window across the source to extract all overlapping tiles.
    /// Optionally generates rotations (90°, 180°, 270°) and reflections
    /// to increase pattern variety. All tiles are deduplicated.
    pub fn extract_tiles(
        source_data: &Array2<usize>,
        tile_size: usize,
        include_rotations: bool,
        include_reflections: bool,
    ) -> Self {
        let (rows, cols) = source_data.dim();

        let mut base_tiles = Vec::new();
        for i in 0..=rows.saturating_sub(tile_size) {
            for j in 0..=cols.saturating_sub(tile_size) {
                let mut tile = [[0; 3]; 3];
                for ti in 0..tile_size {
                    for tj in 0..tile_size {
                        let val = source_data.get((i + ti, j + tj)).copied().unwrap_or(0);
                        if let Some(tile_ref) = tile.get_mut(ti).and_then(|row| row.get_mut(tj)) {
                            *tile_ref = val;
                        }
                    }
                }
                base_tiles.push(tile);
            }
        }

        let all_tiles = if include_rotations || include_reflections {
            let mut transformed_tiles = Vec::new();

            for tile in &base_tiles {
                let mut transforms = vec![*tile];

                if include_rotations {
                    let rot90 = Self::rotate_90(tile);
                    let rot180 = Self::rotate_90(&rot90);
                    let rot270 = Self::rotate_90(&rot180);
                    transforms.push(rot90);
                    transforms.push(rot180);
                    transforms.push(rot270);
                }

                if include_reflections {
                    let current_len = transforms.len();
                    for i in 0..current_len {
                        if let Some(transform) = transforms.get(i) {
                            transforms.push(Self::reflect(transform));
                        }
                    }
                }

                transformed_tiles.extend(transforms);
            }

            Self::deduplicate_tiles(transformed_tiles)
        } else {
            Self::deduplicate_tiles(base_tiles)
        };

        Self {
            source_tiles: all_tiles,
            source_tile_boolean_reference_rules: HashMap::new(),
        }
    }

    fn rotate_90(tile: &Tile) -> Tile {
        let n = 3;
        let mut rotated = [[0; 3]; 3];
        for (i, row) in rotated.iter_mut().enumerate().take(n) {
            for (j, cell) in row.iter_mut().enumerate().take(n) {
                if let Some(tile_row) = tile.get(n - 1 - j)
                    && let Some(&val) = tile_row.get(i)
                {
                    *cell = val;
                }
            }
        }
        rotated
    }

    fn reflect(tile: &Tile) -> Tile {
        let n = 3;
        let mut reflected = [[0; 3]; 3];
        for i in 0..n {
            for j in 0..n {
                if let Some(row) = tile.get(i)
                    && let Some(&val) = row.get(n - 1 - j)
                    && let Some(ref_cell) = reflected.get_mut(i).and_then(|r| r.get_mut(j))
                {
                    *ref_cell = val;
                }
            }
        }
        reflected
    }

    fn deduplicate_tiles(tiles: Vec<Tile>) -> Vec<Tile> {
        let mut seen = HashSet::new();
        let mut unique_tiles = Vec::new();

        for tile in tiles {
            let key: Vec<usize> = tile.iter().flat_map(|row| row.iter().copied()).collect();

            if seen.insert(key) {
                unique_tiles.push(tile);
            }
        }

        unique_tiles
    }

    /// Build boolean reference rules for constraint-based tile selection
    ///
    /// Creates a mapping from boolean constraint patterns to compatible tiles.
    /// Each pattern is a bit vector where 1 means "must contain this cell type"
    /// and 0 means "no constraint". This enables efficient filtering during
    /// wave function collapse propagation.
    pub fn build_boolean_reference_rules(&mut self, unique_cell_count: usize) {
        let mut pattern_to_indices: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();

        // Generate all 2^n possible constraint patterns
        let total_patterns = 1 << unique_cell_count;
        for pattern_bits in 0..total_patterns {
            let pattern: Vec<u8> = (0..unique_cell_count)
                .map(|i| u8::from((pattern_bits >> i) & 1 == 1))
                .collect();

            let mut matching_tiles = Vec::new();
            for (index, tile) in self.source_tiles.iter().enumerate() {
                let tile_i32: [[i32; 3]; 3] =
                    tile.map(|row| row.map(|val| val.try_into().unwrap_or(i32::MAX)));
                let tile_booleans =
                    convert_tile_to_membership_booleans(&tile_i32, unique_cell_count);

                let matches = pattern
                    .iter()
                    .zip(tile_booleans.iter())
                    .all(|(&pattern_bit, &tile_bit)| pattern_bit == 0 || tile_bit == 1);

                if matches {
                    // Tile indices are 1-based (0 reserved for empty)
                    matching_tiles.push(index + 1);
                }
            }

            if !matching_tiles.is_empty() {
                pattern_to_indices.insert(pattern, matching_tiles);
            }
        }

        self.source_tile_boolean_reference_rules = pattern_to_indices;
    }

    /// Calculate exponential sample points for pattern influence decay
    ///
    /// Generates sample points along an exponential decay curve used for
    /// spatial influence calculations. The decay rate is controlled by
    /// the pattern influence distance parameter.
    pub fn calculate_exponential_sample_points(pattern_influence_distance: f64) -> Vec<f64> {
        let k = pattern_influence_distance;
        let step_size = 5.0 * (2.0_f64.ln() / (4.0 * k)).tanh() / 3.0;

        let num_steps = ((0.75 / step_size).ceil() as usize) + 1;
        // Exponential decay formula: k * log(1 - 3x/4) / log(0.5)
        (0..num_steps)
            .map(|i| (i as f64 * step_size).min(0.75))
            .map(|x_val| {
                if x_val == 0.0 {
                    0.0
                } else {
                    k * (1.0 - 3.0 * x_val / 4.0).ln() / (0.5_f64.ln())
                }
            })
            .filter(|&val| val.is_finite())
            .collect()
    }

    /// Get all extracted source tiles
    pub fn source_tiles(&self) -> &[Tile] {
        &self.source_tiles
    }

    /// Get the constraint pattern to compatible tiles mapping
    pub const fn get_boolean_reference_rules(&self) -> &HashMap<Vec<u8>, Vec<usize>> {
        &self.source_tile_boolean_reference_rules
    }
}

/// Convert a tile to membership booleans for constraint matching
///
/// Returns a boolean vector where position i is 1 if the tile contains
/// cell type i+1. Used during wave function collapse to match tiles
/// against constraint patterns.
pub fn convert_tile_to_membership_booleans(
    tile: &[[i32; 3]; 3],
    unique_cell_count: usize,
) -> Vec<u8> {
    let mut unique_values = HashSet::new();
    for row in tile {
        for &val in row {
            if val > 0 {
                unique_values.insert(val as usize);
            }
        }
    }

    // Boolean vector indexed by cell type (1-based)
    (1..=unique_cell_count)
        .map(|i| u8::from(unique_values.contains(&i)))
        .collect()
}

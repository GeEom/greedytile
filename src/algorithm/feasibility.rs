use crate::math::convert::usize_to_f64;
use crate::spatial::tiles::{Tile, convert_tile_to_membership_booleans};
use ndarray::Array2;
use std::collections::BTreeMap;

/// Stores counts of tiles that can legally match each 3x3 region for feasibility scoring
pub struct FeasibilityCountLayer {
    counts: Array2<usize>,
    tile_count: usize,
}

impl FeasibilityCountLayer {
    /// Create a count layer initialized with all tiles feasible at each position
    #[must_use]
    pub fn new(rows: usize, cols: usize, tile_count: usize) -> Self {
        Self {
            counts: Array2::from_elem((rows, cols), tile_count),
            tile_count,
        }
    }

    /// Count the source tiles that match `tile_grid`, whose zeros match anything
    pub fn update_count(
        &mut self,
        row: usize,
        col: usize,
        tile_grid: &[[i32; 3]; 3],
        source_tiles: &[Tile],
        dispatch_rules: &BTreeMap<Vec<u8>, Vec<usize>>,
        unique_cell_count: usize,
    ) {
        let tile_booleans = convert_tile_to_membership_booleans(tile_grid, unique_cell_count);
        let potential_sources = dispatch_rules
            .get(&tile_booleans)
            .map_or(&[][..], Vec::as_slice);

        let tile_pattern: [[i32; 3]; 3] =
            tile_grid.map(|tile_row| tile_row.map(|val| if val == 0 { -1 } else { val }));

        let mut count = 0;
        for &ref_index in potential_sources {
            if ref_index > 0 && ref_index <= source_tiles.len() {
                let Some(source_tile) = source_tiles.get(ref_index - 1) else {
                    continue;
                };

                let matches =
                    tile_pattern
                        .iter()
                        .zip(source_tile.iter())
                        .all(|(pattern_row, source_row)| {
                            pattern_row.iter().zip(source_row.iter()).all(
                                |(&pattern_val, &source_val)| {
                                    pattern_val == -1
                                        || pattern_val == source_val.try_into().unwrap_or(i32::MAX)
                                },
                            )
                        });

                if matches {
                    count += 1;
                }
            }
        }

        if let Some(count_ref) = self.counts.get_mut([row, col]) {
            *count_ref = count;
        }
    }

    /// The fraction of tiles still feasible here; 1.0 outside the counted area
    #[must_use]
    pub fn get_fraction(&self, row: usize, col: usize) -> f64 {
        if row < self.counts.nrows() && col < self.counts.ncols() {
            usize_to_f64(
                self.counts
                    .get([row, col])
                    .copied()
                    .unwrap_or(self.tile_count),
            ) / usize_to_f64(self.tile_count)
        } else {
            1.0
        }
    }

    /// Resize, keeping existing counts and giving new cells every tile
    pub fn extend_to(&mut self, new_rows: usize, new_cols: usize) {
        if new_rows == self.counts.nrows() && new_cols == self.counts.ncols() {
            return;
        }

        let mut new_counts = Array2::from_elem((new_rows, new_cols), self.tile_count);
        for i in 0..self.counts.nrows().min(new_rows) {
            for j in 0..self.counts.ncols().min(new_cols) {
                if let Some(count) = self.counts.get([i, j]).copied()
                    && let Some(new_count) = new_counts.get_mut([i, j])
                {
                    *new_count = count;
                }
            }
        }
        self.counts = new_counts;
    }
}

use crate::spatial::tiles::{Tile, convert_tile_to_membership_booleans};
use ndarray::Array2;
use std::collections::HashMap;

/// Stores counts of tiles that can legally match each 3x3 region for feasibility scoring
pub struct FeasibilityCountLayer {
    counts: Array2<usize>,
    tile_count: usize,
}

impl FeasibilityCountLayer {
    /// Create a count layer initialized with all tiles feasible at each position
    pub fn new(rows: usize, cols: usize, tile_count: usize) -> Self {
        Self {
            counts: Array2::from_elem((rows, cols), tile_count),
            tile_count,
        }
    }

    /// Update the feasible tile count for a 3x3 region centered at (row, col)
    ///
    /// Matches the `tile_grid` pattern against source tiles using dispatch rules
    /// to determine which tiles are compatible with the current constraints
    pub fn update_count(
        &mut self,
        row: usize,
        col: usize,
        tile_grid: &[[i32; 3]; 3],
        source_tiles: &[Tile],
        dispatch_rules: &HashMap<Vec<u8>, Vec<usize>>,
        unique_cell_count: usize,
    ) {
        let tile_booleans = convert_tile_to_membership_booleans(tile_grid, unique_cell_count);
        let potential_sources = dispatch_rules
            .get(&tile_booleans)
            .cloned()
            .unwrap_or_default();

        let tile_pattern: [[i32; 3]; 3] =
            tile_grid.map(|tile_row| tile_row.map(|val| if val == 0 { -1 } else { val }));

        let mut count = 0;
        for &ref_index in &potential_sources {
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

    /// Returns the fraction of tiles that remain feasible at this position
    ///
    /// Used to calculate feasibility scores where 1.0 means all tiles are possible
    /// and values approaching 0.0 indicate highly constrained positions
    pub fn get_fraction(&self, row: usize, col: usize) -> f64 {
        if row < self.counts.nrows() && col < self.counts.ncols() {
            self.counts
                .get([row, col])
                .copied()
                .unwrap_or(self.tile_count) as f64
                / self.tile_count as f64
        } else {
            1.0
        }
    }

    /// Resize the count array while preserving existing data
    ///
    /// New positions are initialized with full feasibility (all tiles viable)
    pub fn extend_to(&mut self, new_rows: usize, new_cols: usize) {
        if new_rows == self.counts.nrows() && new_cols == self.counts.ncols() {
            return;
        }

        let mut new_counts = Array2::from_elem((new_rows, new_cols), self.tile_count);
        for i in 0..self.counts.nrows().min(new_rows) {
            for j in 0..self.counts.ncols().min(new_cols) {
                if let Some(count) = self.counts.get([i, j]).copied()
                    && let Some(new_count) = new_counts.get_mut([i, j]) {
                        *new_count = count;
                    }
            }
        }
        self.counts = new_counts;
    }
}

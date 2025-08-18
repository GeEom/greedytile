use crate::{
    algorithm::cache::ViableTilesCache,
    algorithm::feasibility::FeasibilityCountLayer,
    algorithm::selection::compute_viable_tiles_at_position,
    io::configuration::ADJACENCY_LEVELS,
    io::visualization::VisualizationCapture,
    spatial::tiles::Tile,
    spatial::{GridState, grid},
};
use ndarray::{Array4, ArrayView3};
use std::collections::{HashMap, VecDeque};
use std::ops::Range;

/// Algorithm parameters and source data that remain constant across iterations
pub struct StepData {
    /// Frequency ratios for each tile type in the source
    pub source_ratios: Vec<f64>,
    /// Total number of unique tile types
    pub unique_cell_count: usize,
    /// Radius for grid extension operations
    pub grid_extension_radius: i32,
    /// Threshold for density correction activation
    pub density_correction_threshold: f64,
    /// Steepness of density correction sigmoid
    pub density_correction_steepness: f64,
    /// Minimum strength for density correction
    pub density_minimum_strength: f64,
    /// All unique tile patterns extracted from source
    pub source_tiles: Vec<Tile>,
    /// Mapping from constraint patterns to compatible tiles
    pub tile_compatibility_rules: HashMap<Vec<u8>, Vec<usize>>,
}

/// A rectangular region defined by row and column ranges
pub struct Region {
    /// Row indices range
    pub rows: Range<usize>,
    /// Column indices range
    pub cols: Range<usize>,
}

impl Region {
    /// Create a new region from row and column ranges
    pub const fn new(rows: Range<usize>, cols: Range<usize>) -> Self {
        Self { rows, cols }
    }

    /// Get a clone of the row range
    pub fn rows(&self) -> Range<usize> {
        self.rows.clone()
    }

    /// Get a clone of the column range
    pub fn cols(&self) -> Range<usize> {
        self.cols.clone()
    }
}

/// Apply probability influence matrix and recalculate entropy after placing a tile
///
/// Updates the region around the selected position based on the tile's
/// influence patterns from the precomputed probability influence matrices
pub fn update_probabilities_and_entropy(
    grid_state: &mut GridState,
    probability_influence_matrices: &Array4<f64>,
    selected_cell_reference: usize,
    selection_coordinates: [i32; 2],
    system_offset: [i32; 2],
    step_data: &StepData,
) {
    let (row_span, col_span) = grid::get_region_spans(
        &system_offset,
        &selection_coordinates,
        step_data.grid_extension_radius,
    );

    let row_start = row_span.start.min(grid_state.rows());
    let row_end = row_span.end.min(grid_state.rows());
    let col_start = col_span.start.min(grid_state.cols());
    let col_end = col_span.end.min(grid_state.cols());

    let region = Region::new(row_start..row_end, col_start..col_end);
    let impact =
        probability_influence_matrices.index_axis(ndarray::Axis(0), selected_cell_reference - 1);

    // Fused update reduces memory traversals from 2N to N
    update_probabilities_and_entropy_fused(grid_state, &impact, &region);
}

/// Update probabilities and entropy in a single pass over the affected region
///
/// Applies the influence matrix to tile probabilities and immediately
/// recalculates entropy using mean normalization to avoid separate traversals
pub fn update_probabilities_and_entropy_fused(
    grid_state: &mut GridState,
    impact: &ArrayView3<'_, f64>,
    region: &Region,
) {
    for (i, row) in region.rows().enumerate() {
        for (j, col) in region.cols().enumerate() {
            let mut sum = 0.0;

            for color in 0..grid_state.unique_cell_count {
                let impact_val = impact.get([color, i, j]).copied().unwrap_or(1.0);
                if let Some(tile_probs) = grid_state.tile_probabilities.get_mut(color)
                    && let Some(prob) = tile_probs.get_mut([row, col]) {
                        *prob *= impact_val;
                    }
            }

            for color in 0..grid_state.unique_cell_count {
                if let Some(tile_probs) = grid_state.tile_probabilities.get(color)
                    && let Some(prob) = tile_probs.get([row, col]) {
                        sum += prob;
                    }
            }

            // Mean normalization prevents numerical instability in entropy calculation
            let mean_prob = sum / grid_state.unique_cell_count as f64;
            let entropy = if mean_prob > 0.0 {
                let mut entropy_sum = 0.0;
                for color in 0..grid_state.unique_cell_count {
                    let p = grid_state
                        .tile_probabilities
                        .get(color)
                        .and_then(|probs| probs.get([row, col]))
                        .copied()
                        .unwrap_or(0.0)
                        / mean_prob;
                    if p > 0.0 {
                        entropy_sum += p * p.ln();
                    }
                }
                entropy_sum
            } else {
                0.0
            };
            if let Some(entropy_val) = grid_state.entropy.get_mut([row, col]) {
                *entropy_val = entropy;
            }
        }
    }
}

/// Mark the selected tile position as locked and update adjacency weights
///
/// Adjacency weights decrease with distance to guide future tile selection
/// toward positions near already-placed tiles
pub fn update_grid_state(
    grid_state: &mut GridState,
    selected_cell_reference: usize,
    selection_coordinates: [i32; 2],
    system_offset: [i32; 2],
    visualization: &mut Option<VisualizationCapture>,
    iteration: usize,
) {
    for level in 1..=ADJACENCY_LEVELS {
        let weight_increment = (1 + ADJACENCY_LEVELS - level) as u32;
        let (row_span, col_span) =
            grid::get_region_spans(&system_offset, &selection_coordinates, level as i32);
        for row in row_span {
            for col in col_span.clone() {
                if let Some(weight) = grid_state.adjacency_weights.get_mut([row, col]) {
                    *weight += weight_increment;
                }
            }
        }
    }

    let (row_span_0, col_span_0) =
        grid::get_region_spans(&system_offset, &selection_coordinates, 0);
    for row in row_span_0 {
        for col in col_span_0.clone() {
            if let Some(locked) = grid_state.locked_tiles.get_mut([row, col]) {
                *locked += selected_cell_reference as u32;

                if let Some(viz) = visualization {
                    let abs_row = row as i32 - system_offset[0];
                    let abs_col = col as i32 - system_offset[1];
                    viz.record_placement(abs_row, abs_col, *locked, iteration);
                }
            }
        }
    }
}

/// Position with exactly one compatible tile based on surrounding constraints
#[derive(Debug, Clone)]
pub struct ForcedPosition {
    /// World coordinates of the forced position
    pub coordinates: [i32; 2],
    /// The single tile that can be placed here
    pub tile_reference: usize,
}

/// Find adjacent positions that are forced to a single tile choice
///
/// These positions can be filled immediately without selection logic,
/// potentially triggering cascades of forced placements
pub fn detect_forced_positions(
    grid_state: &GridState,
    position: [i32; 2],
    system_offset: [i32; 2],
    source_tiles: &[Tile],
    step_data: &StepData,
    cache: &mut crate::algorithm::cache::ViableTilesCache,
) -> Vec<ForcedPosition> {
    let mut forced = Vec::new();

    for di in -1..=1 {
        for dj in -1..=1 {
            if di == 0 && dj == 0 {
                continue;
            }

            let check_pos = [position[0] + di, position[1] + dj];

            // Skip positions outside bounds
            if let Some(bounds) = &grid_state.generation_bounds
                && !bounds.contains(check_pos) {
                    continue;
                }

            let row = (check_pos[0] + system_offset[0]) as usize;
            let col = (check_pos[1] + system_offset[1]) as usize;
            if row >= grid_state.rows() || col >= grid_state.cols() {
                continue;
            }
            if grid_state
                .locked_tiles
                .get([row, col])
                .copied()
                .unwrap_or(0)
                > 1
            {
                continue;
            }

            let viable = crate::algorithm::selection::compute_viable_tiles_at_position(
                grid_state,
                check_pos,
                system_offset,
                source_tiles,
                step_data,
                cache,
            );

            if viable.len() == 1
                && let Some(&tile_ref) = viable.first() {
                    forced.push(ForcedPosition {
                        coordinates: check_pos,
                        tile_reference: tile_ref,
                    });
                }
        }
    }

    forced
}

/// Pipeline for processing positions with only one viable tile option
#[derive(Debug)]
pub struct ForcedPipeline {
    /// Queue of positions that must be filled with specific tiles
    pub queue: VecDeque<ForcedPosition>,
}

impl Default for ForcedPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl ForcedPipeline {
    /// Create a new empty pipeline
    pub const fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }

    /// Add forced positions to the queue, skipping any already present
    pub fn add_positions(&mut self, positions: Vec<ForcedPosition>) {
        for pos in positions {
            if !self.queue.iter().any(|p| p.coordinates == pos.coordinates) {
                self.queue.push_back(pos);
            }
        }
    }

    /// Remove and return the next forced position
    pub fn take_next(&mut self) -> Option<ForcedPosition> {
        self.queue.pop_front()
    }

    /// Check if the pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Get the number of pending forced positions
    pub fn len(&self) -> usize {
        self.queue.len()
    }
}

/// Detect positions that have adjacent tiles but no compatible options
///
/// Returns the first contradiction found, which indicates the algorithm
/// has reached an unsolvable state requiring backtracking or restart
pub fn check_for_contradiction(
    grid_state: &GridState,
    system_offset: [i32; 2],
    step_data: &StepData,
    cache: &mut ViableTilesCache,
) -> Option<[usize; 2]> {
    for i in 0..grid_state.rows() {
        for j in 0..grid_state.cols() {
            if grid_state.locked_tiles.get([i, j]).copied().unwrap_or(0) > 1 {
                continue;
            }

            if grid_state
                .adjacency_weights
                .get([i, j])
                .copied()
                .unwrap_or(0)
                > 1
            {
                let coords = [i as i32 - system_offset[0], j as i32 - system_offset[1]];
                let viable = compute_viable_tiles_at_position(
                    grid_state,
                    coords,
                    system_offset,
                    &step_data.source_tiles,
                    step_data,
                    cache,
                );

                if viable.is_empty() {
                    return Some([i, j]);
                }
            }
        }
    }
    None
}

/// Recalculate feasibility scores in the region affected by tile placement
///
/// Feasibility represents how many source tiles can match each position,
/// used to identify highly constrained areas that need priority attention
pub fn update_feasibility_counts(
    grid_state: &mut GridState,
    feasibility_layer: &mut FeasibilityCountLayer,
    selection_coordinates: [i32; 2],
    system_offset: [i32; 2],
    step_data: &StepData,
) {
    let (row_span, col_span) = grid::get_region_spans(
        &system_offset,
        &selection_coordinates,
        ADJACENCY_LEVELS as i32,
    );

    for source_row in row_span.clone() {
        for source_col in col_span.clone() {
            if source_row + 2 < grid_state.rows() && source_col + 2 < grid_state.cols() {
                let mut tile_grid = [[0i32; 3]; 3];

                for di in 0..3 {
                    for dj in 0..3 {
                        let grid_row = source_row + di;
                        let grid_col = source_col + dj;

                        if grid_row < grid_state.rows() && grid_col < grid_state.cols() {
                            let locked_val = grid_state
                                .locked_tiles
                                .get([grid_row, grid_col])
                                .copied()
                                .unwrap_or(0);
                            if locked_val > 0
                                && let Some(tile_ref) =
                                    tile_grid.get_mut(di).and_then(|row| row.get_mut(dj))
                                {
                                    *tile_ref = (locked_val - 1) as i32;
                                }
                        }
                    }
                }

                feasibility_layer.update_count(
                    source_row,
                    source_col,
                    &tile_grid,
                    &step_data.source_tiles,
                    &step_data.tile_compatibility_rules,
                    step_data.unique_cell_count,
                );
            }
        }
    }

    // Average feasibility from all overlapping 3x3 regions
    let target_row_start = (row_span.start + 1).min(grid_state.rows());
    let target_row_end = row_span.end.min(grid_state.rows());
    let target_col_start = (col_span.start + 1).min(grid_state.cols());
    let target_col_end = col_span.end.min(grid_state.cols());

    for target_row in target_row_start..target_row_end {
        for target_col in target_col_start..target_col_end {
            let mut feasibility_sum = 0.0;
            let mut count = 0;

            for dr in -1..=1 {
                for dc in -1..=1 {
                    let src_row = (target_row as i32 + dr) as usize;
                    let src_col = (target_col as i32 + dc) as usize;

                    if src_row < grid_state.rows() && src_col < grid_state.cols() {
                        feasibility_sum += feasibility_layer.get_fraction(src_row, src_col);
                        count += 1;
                    }
                }
            }

            if count > 0
                && let Some(feas) = grid_state
                    .feasibility
                    .get_mut([target_row + 1, target_col + 1])
                {
                    *feas = feasibility_sum / count as f64;
                }
        }
    }
}

use crate::{
    algorithm::{feasibility::FeasibilityCountLayer, propagation::StepData},
    io::{
        configuration::{ADJACENCY_LEVELS, BASE_REMOVAL_RADIUS, MAX_REMOVAL_RADIUS},
        visualization::VisualizationCapture,
    },
    math::convert::{usize_to_i32, usize_to_u32},
    spatial::{GridState, grid},
};
use ndarray::Array4;
use std::ops::Range;

/// Summary of tiles affected by deadlock resolution
#[derive(Debug)]
pub struct DeadlockResolutionResult {
    /// Total number of tiles that were unlocked
    pub tiles_unlocked: usize,
    /// Grid positions of all unlocked tiles
    pub unlocked_positions: Vec<[usize; 2]>,
}

/// The state a deadlock resolution rewinds
pub struct DeadlockState<'a> {
    /// Grid whose tiles, weights, probabilities and scores are rewound
    pub grid_state: &'a mut GridState,
    /// Per-region counts of tiles that still fit
    pub feasibility_layer: &'a mut FeasibilityCountLayer,
    /// Placements per tile type, decremented for each removed tile
    pub selection_tally: &'a mut [usize],
    /// Placement recording, which logs each removal when present
    pub visualization: &'a mut Option<VisualizationCapture>,
}

/// Unlock the tiles around a contradiction, clearing a wider area each time
/// the same cell deadlocks so the search cannot loop
#[must_use]
pub fn resolve_spatial_deadlock(
    state: DeadlockState<'_>,
    contradiction_pos: [usize; 2],
    system_offset: [i32; 2],
    step_data: &StepData,
    probability_influence_matrices: &Array4<f64>,
    iteration: usize,
) -> DeadlockResolutionResult {
    let DeadlockState {
        grid_state,
        feasibility_layer,
        selection_tally,
        visualization,
    } = state;

    // Each repeat at this location widens the next removal
    if let Some(count) = grid_state.removal_count.get_mut(contradiction_pos) {
        *count = count.saturating_add(1);
    }

    let contradiction_coords = [
        usize_to_i32(contradiction_pos[0]) - system_offset[0],
        usize_to_i32(contradiction_pos[1]) - system_offset[1],
    ];

    let removal_count = grid_state
        .removal_count
        .get(contradiction_pos)
        .copied()
        .unwrap_or(0);

    let removal_radius = (BASE_REMOVAL_RADIUS + i32::from(removal_count)).min(MAX_REMOVAL_RADIUS);

    let (row_span, col_span) =
        grid::get_region_spans(system_offset, contradiction_coords, removal_radius);

    let mut tiles_to_unlock = Vec::new();
    for row in row_span {
        for col in col_span.clone() {
            let locked_val = grid_state
                .locked_tiles
                .get([row, col])
                .copied()
                .unwrap_or(0);
            if locked_val > 1 {
                tiles_to_unlock.push((row, col, locked_val - 1));
            }
        }
    }

    for &(row, col, tile_reference) in &tiles_to_unlock {
        let coords = [
            usize_to_i32(row) - system_offset[0],
            usize_to_i32(col) - system_offset[1],
        ];

        if let Some(tile_matrix) = grid_state.locked_tiles.get_mut([row, col]) {
            *tile_matrix = tile_matrix.saturating_sub(tile_reference);

            if let Some(viz) = visualization.as_mut() {
                viz.record_removal(coords[0], coords[1], iteration);
            }
        }

        // Tile reference 2+ maps to tally index 0+
        if tile_reference >= 1
            && let Some(tally) = selection_tally.get_mut(tile_reference as usize - 1)
        {
            *tally = tally.saturating_sub(1);
        }

        revert_adjacency_weights(grid_state, system_offset, coords);
        revert_probabilities(
            grid_state,
            probability_influence_matrices,
            tile_reference as usize,
            system_offset,
            coords,
            step_data,
        );
    }

    // Recalculate entropy in affected region with expanded radius
    let entropy_radius = step_data.grid_extension_radius + removal_radius;
    let (entropy_row_span, entropy_col_span) =
        grid::get_region_spans(system_offset, contradiction_coords, entropy_radius);
    recompute_entropy(grid_state, step_data, entropy_row_span, entropy_col_span);

    // Update feasibility counts in the extended region
    let feasibility_update_radius = (usize_to_i32(ADJACENCY_LEVELS) + 1) + removal_radius;
    let (feas_row_span, feas_col_span) = grid::get_region_spans(
        system_offset,
        contradiction_coords,
        feasibility_update_radius,
    );
    recount_feasibility(
        grid_state,
        feasibility_layer,
        step_data,
        &feas_row_span,
        &feas_col_span,
    );
    aggregate_feasibility(grid_state, feasibility_layer, feas_row_span, feas_col_span);

    DeadlockResolutionResult {
        tiles_unlocked: tiles_to_unlock.len(),
        unlocked_positions: tiles_to_unlock
            .iter()
            .map(|&(row, col, _)| [row, col])
            .collect(),
    }
}

/// Take back the adjacency weight a removed tile added at every level
fn revert_adjacency_weights(grid_state: &mut GridState, system_offset: [i32; 2], coords: [i32; 2]) {
    for level in 1..=ADJACENCY_LEVELS {
        let weight_decrement = usize_to_u32(1 + ADJACENCY_LEVELS - level);
        let (adj_row_span, adj_col_span) =
            grid::get_region_spans(system_offset, coords, usize_to_i32(level));

        for adj_row in adj_row_span {
            for adj_col in adj_col_span.clone() {
                if let Some(weight) = grid_state.adjacency_weights.get_mut([adj_row, adj_col]) {
                    *weight = weight.saturating_sub(weight_decrement);
                }
            }
        }
    }
}

/// Divide out the probability influence a removed tile multiplied in
fn revert_probabilities(
    grid_state: &mut GridState,
    probability_influence_matrices: &Array4<f64>,
    tile_reference: usize,
    system_offset: [i32; 2],
    coords: [i32; 2],
    step_data: &StepData,
) {
    let n_tiles = probability_influence_matrices
        .shape()
        .first()
        .copied()
        .unwrap_or(0);
    if tile_reference == 0 || tile_reference > n_tiles {
        return;
    }

    let (prob_row_span, prob_col_span) =
        grid::get_region_spans(system_offset, coords, step_data.grid_extension_radius);

    let impact = probability_influence_matrices.index_axis(ndarray::Axis(0), tile_reference - 1);
    let impact_rows = impact.shape().get(1).copied().unwrap_or(0);
    let impact_cols = impact.shape().get(2).copied().unwrap_or(0);

    let row_start = prob_row_span.start.min(grid_state.rows());
    let row_end = prob_row_span.end.min(grid_state.rows());
    let col_start = prob_col_span.start.min(grid_state.cols());
    let col_end = prob_col_span.end.min(grid_state.cols());

    for (i, row_index) in (row_start..row_end).enumerate() {
        for (j, col_index) in (col_start..col_end).enumerate() {
            if i >= impact_rows || j >= impact_cols {
                continue;
            }

            for color in 0..step_data.unique_cell_count {
                let impact_value = impact.get([color, i, j]).copied().unwrap_or(1.0);
                if impact_value != 0.0
                    && color < grid_state.tile_probabilities.len()
                    && row_index < grid_state.rows()
                    && col_index < grid_state.cols()
                    && let Some(prob_matrix) = grid_state.tile_probabilities.get_mut(color)
                    && let Some(prob) = prob_matrix.get_mut([row_index, col_index])
                {
                    *prob /= impact_value;
                }
            }
        }
    }
}

/// Recompute the entropy of every unlocked cell in the given spans
fn recompute_entropy(
    grid_state: &mut GridState,
    step_data: &StepData,
    row_span: Range<usize>,
    col_span: Range<usize>,
) {
    for row in row_span.start..row_span.end.min(grid_state.rows()) {
        for col in col_span.start..col_span.end.min(grid_state.cols()) {
            if grid_state
                .locked_tiles
                .get([row, col])
                .copied()
                .unwrap_or(0)
                > 1
            {
                continue;
            }

            let mut sum = 0.0;
            let mut count = 0;

            for color in 0..step_data.unique_cell_count {
                if let Some(prob) = grid_state
                    .tile_probabilities
                    .get(color)
                    .and_then(|p| p.get([row, col]))
                    .copied()
                {
                    sum += prob;
                    count += 1;
                }
            }

            let mean_prob = if count > 0 {
                sum / f64::from(count)
            } else {
                0.0
            };
            let entropy = if mean_prob > 0.0 {
                let mut entropy_sum = 0.0;

                for color in 0..step_data.unique_cell_count {
                    if let Some(prob) = grid_state
                        .tile_probabilities
                        .get(color)
                        .and_then(|p| p.get([row, col]))
                        .copied()
                    {
                        let normalized = prob / mean_prob;
                        if normalized > 0.0 {
                            let term = normalized * normalized.ln();
                            entropy_sum += term;
                        }
                    }
                }

                entropy_sum
            } else {
                0.0
            };

            if row < grid_state.rows()
                && col < grid_state.cols()
                && let Some(entropy_val) = grid_state.entropy.get_mut([row, col])
            {
                *entropy_val = entropy;
            }
        }
    }
}

/// Recount the tiles that fit each 3x3 region starting in the given spans
fn recount_feasibility(
    grid_state: &GridState,
    feasibility_layer: &mut FeasibilityCountLayer,
    step_data: &StepData,
    row_span: &Range<usize>,
    col_span: &Range<usize>,
) {
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
                                *tile_ref = (locked_val - 1).cast_signed();
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
}

/// Score each cell by the mean squared fraction of the regions around it
fn aggregate_feasibility(
    grid_state: &mut GridState,
    feasibility_layer: &FeasibilityCountLayer,
    row_span: Range<usize>,
    col_span: Range<usize>,
) {
    for target_row in row_span.start..row_span.end.min(grid_state.rows()) {
        for target_col in col_span.start..col_span.end.min(grid_state.cols()) {
            let mut feasibility_sum = 0.0;
            let mut count = 0;

            for dr in -1..=1 {
                for dc in -1..=1 {
                    if let (Some(src_row), Some(src_col)) = (
                        target_row.checked_add_signed(dr - 1),
                        target_col.checked_add_signed(dc - 1),
                    ) && src_row < grid_state.rows()
                        && src_col < grid_state.cols()
                    {
                        let fraction = feasibility_layer.get_fraction(src_row, src_col);
                        let squared = fraction * fraction;
                        feasibility_sum += squared;
                        count += 1;
                    }
                }
            }

            if count > 0
                && let Some(feas) = grid_state.feasibility.get_mut([target_row, target_col])
            {
                *feas = feasibility_sum / f64::from(count);
            }
        }
    }
}

use crate::{
    algorithm::{feasibility::FeasibilityCountLayer, propagation::StepData},
    io::{
        configuration::{ADJACENCY_LEVELS, BASE_REMOVAL_RADIUS, MAX_REMOVAL_RADIUS},
        visualization::VisualizationCapture,
    },
    spatial::{GridState, grid},
};

/// Summary of tiles affected by deadlock resolution
#[derive(Debug)]
pub struct DeadlockResolutionResult {
    /// Total number of tiles that were unlocked
    pub tiles_unlocked: usize,
    /// Grid positions of all unlocked tiles
    pub unlocked_positions: Vec<[usize; 2]>,
}

/// Resolve a spatial deadlock by unlocking tiles around the contradiction
///
/// Uses an adaptive radius that expands with repeated deadlocks at the same
/// location. This prevents the algorithm from getting stuck in loops by
/// progressively clearing larger areas when contradictions persist.
pub fn resolve_spatial_deadlock(
    grid_state: &mut GridState,
    feasibility_layer: &mut FeasibilityCountLayer,
    contradiction_pos: [usize; 2],
    system_offset: [i32; 2],
    selection_tally: &mut [usize],
    step_data: &StepData,
    probability_influence_matrices: &ndarray::Array4<f64>,
    visualization: &mut Option<VisualizationCapture>,
    iteration: usize,
) -> DeadlockResolutionResult {
    // Add one to the removal count for this location, to increases the radius of removal on repeated trigger
    if let Some(count) = grid_state.removal_count.get_mut(contradiction_pos) {
        *count = count.saturating_add(1);
    }

    let contradiction_coords = [
        contradiction_pos[0] as i32 - system_offset[0],
        contradiction_pos[1] as i32 - system_offset[1],
    ];

    let mut unlocked_positions = Vec::new();
    let mut tiles_unlocked = 0;

    // Adaptive radius prevents repeated deadlocks at the same location
    let removal_count = grid_state
        .removal_count
        .get(contradiction_pos)
        .copied()
        .unwrap_or(0);

    let removal_radius = (BASE_REMOVAL_RADIUS + removal_count as i32).min(MAX_REMOVAL_RADIUS);

    let (row_span, col_span) =
        grid::get_region_spans(&system_offset, &contradiction_coords, removal_radius);

    let mut tiles_to_unlock = Vec::new();

    for row in row_span {
        for col in col_span.clone() {
            let locked_val = grid_state
                .locked_tiles
                .get([row, col])
                .copied()
                .unwrap_or(0);
            if locked_val > 1 {
                let tile_reference = locked_val - 1;
                tiles_to_unlock.push((row, col, tile_reference));
                unlocked_positions.push([row, col]);
                tiles_unlocked += 1;
            }
        }
    }

    // Reverse the effects of placing each locked tile
    for (row, col, tile_reference) in tiles_to_unlock {
        if let Some(tile_matrix) = grid_state.locked_tiles.get_mut([row, col]) {
            *tile_matrix = tile_matrix.saturating_sub(tile_reference);

            if let Some(viz) = visualization {
                let abs_row = row as i32 - system_offset[0];
                let abs_col = col as i32 - system_offset[1];
                viz.record_removal(abs_row, abs_col, iteration);
            }
        }

        // Decrement tally for non-empty tiles (tile_reference 2+ maps to tally index 0+)
        if tile_reference >= 1
            && let Some(tally) = selection_tally.get_mut(tile_reference as usize - 1) {
                *tally = tally.saturating_sub(1);
            }

        // Revert adjacency weights for all affected levels
        let coords = [row as i32 - system_offset[0], col as i32 - system_offset[1]];

        for level in 1..=ADJACENCY_LEVELS {
            let weight_decrement = (1 + ADJACENCY_LEVELS - level) as u32;
            let (adj_row_span, adj_col_span) =
                grid::get_region_spans(&system_offset, &coords, level as i32);

            for adj_row in adj_row_span {
                for adj_col in adj_col_span.clone() {
                    if let Some(weight) = grid_state.adjacency_weights.get_mut([adj_row, adj_col]) {
                        *weight = weight.saturating_sub(weight_decrement);
                    }
                }
            }
        }

        // Reverse probability mutations by dividing out the influence values
        let n_tiles = probability_influence_matrices
            .shape()
            .first()
            .copied()
            .unwrap_or(0);
        if tile_reference == 0 || tile_reference as usize > n_tiles {
            continue;
        }

        let influence_radius = step_data.grid_extension_radius;
        let (prob_row_span, prob_col_span) =
            grid::get_region_spans(&system_offset, &coords, influence_radius);

        let impact = probability_influence_matrices
            .index_axis(ndarray::Axis(0), tile_reference as usize - 1);

        let impact_shape = impact.shape();

        let row_start = prob_row_span.start.min(grid_state.rows());
        let row_end = prob_row_span.end.min(grid_state.rows());
        let col_start = prob_col_span.start.min(grid_state.cols());
        let col_end = prob_col_span.end.min(grid_state.cols());

        for (i, row_index) in (row_start..row_end).enumerate() {
            for (j, col_index) in (col_start..col_end).enumerate() {
                if i >= impact_shape.get(1).copied().unwrap_or(0)
                    || j >= impact_shape.get(2).copied().unwrap_or(0)
                {
                    continue;
                }

                // Divide out the influence for all tile types at this position
                for color in 0..step_data.unique_cell_count {
                    let impact_value = impact.get([color, i, j]).copied().unwrap_or(1.0);
                    if impact_value != 0.0
                        && color < grid_state.tile_probabilities.len()
                        && row_index < grid_state.rows()
                        && col_index < grid_state.cols()
                        && let Some(prob_matrix) = grid_state.tile_probabilities.get_mut(color)
                            && let Some(prob) = prob_matrix.get_mut([row_index, col_index]) {
                                *prob /= impact_value;
                            }
                }
            }
        }
    }

    // Recalculate entropy in affected region with expanded radius
    let entropy_radius = step_data.grid_extension_radius + removal_radius;
    let (entropy_row_span, entropy_col_span) =
        grid::get_region_spans(&system_offset, &contradiction_coords, entropy_radius);

    for row in entropy_row_span.start..entropy_row_span.end.min(grid_state.rows()) {
        for col in entropy_col_span.start..entropy_col_span.end.min(grid_state.cols()) {
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

            let mean_prob = if count > 0 { sum / count as f64 } else { 0.0 };
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
                            entropy_sum += normalized * normalized.ln();
                        }
                    }
                }

                entropy_sum
            } else {
                0.0
            };

            if row < grid_state.rows() && col < grid_state.cols()
                && let Some(entropy_val) = grid_state.entropy.get_mut([row, col]) {
                    *entropy_val = entropy;
                }
        }
    }

    // Update feasibility counts in the extended region
    let feasibility_update_radius = (ADJACENCY_LEVELS as i32 + 1) + removal_radius;
    let (feas_row_span, feas_col_span) = grid::get_region_spans(
        &system_offset,
        &contradiction_coords,
        feasibility_update_radius,
    );

    for source_row in feas_row_span.clone() {
        for source_col in feas_col_span.clone() {
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

    // Aggregate feasibility scores from surrounding pattern counts
    for target_row in feas_row_span.start..feas_row_span.end.min(grid_state.rows()) {
        for target_col in feas_col_span.start..feas_col_span.end.min(grid_state.cols()) {
            let mut feasibility_sum = 0.0;
            let mut count = 0;

            for dr in -1..=1 {
                for dc in -1..=1 {
                    let src_row = (target_row as i32 + dr - 1) as usize;
                    let src_col = (target_col as i32 + dc - 1) as usize;

                    if src_row < grid_state.rows() && src_col < grid_state.cols() {
                        let fraction = feasibility_layer.get_fraction(src_row, src_col);
                        feasibility_sum += fraction * fraction;
                        count += 1;
                    }
                }
            }

            if count > 0
                && let Some(feas) = grid_state.feasibility.get_mut([target_row, target_col]) {
                    *feas = feasibility_sum / count as f64;
                }
        }
    }

    DeadlockResolutionResult {
        tiles_unlocked,
        unlocked_positions,
    }
}

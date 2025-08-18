use crate::{
    algorithm::{
        bitset::TileBitset,
        cache::{PatternKey, ViableTilesCache},
        propagation::StepData,
    },
    math::probability::erf,
    spatial::tiles::{Tile, convert_tile_to_membership_booleans},
    spatial::{GridState, grid},
};
use std::collections::HashMap;

// Algorithm-specific constants for position and tile selection
/// Number of top adjacency candidates to consider for selection
pub const ADJACENCY_CANDIDATES_CONSIDERED: usize = 20;
/// Number of top candidates to consider for final selection
pub const CANDIDATES_CONSIDERED: usize = 15;

/// Determine which tiles can be legally placed at the given position
///
/// Uses bitset intersection for efficiency and caches pattern lookups.
/// Checks positions in order of expected constraint strength for early termination.
pub fn compute_viable_tiles_at_position(
    grid_state: &GridState,
    position: [i32; 2],
    system_offset: [i32; 2],
    source_tiles: &[Tile],
    step_data: &StepData,
    cache: &mut ViableTilesCache,
) -> Vec<usize> {
    // Center position typically provides strongest constraints
    let positions = [
        (1, 1),
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (0, 0),
        (0, 2),
        (2, 0),
        (2, 2),
    ];

    let mut result_bitset: Option<TileBitset> = None;

    for (i, j) in positions {
        let (row_span, col_span) = grid::get_region_spans(
            &system_offset,
            &[position[0] + i as i32 - 1, position[1] + j as i32 - 1],
            1,
        );
        if (row_span.end - row_span.start <= 2) || (col_span.end - col_span.start <= 2) {
            continue;
        }
        let mut tile_3x3 = [[0i32; 3]; 3];

        for di in 0..3 {
            for dj in 0..3 {
                let r = row_span.start + di;
                let c = col_span.start + dj;
                if r < grid_state.rows() && c < grid_state.cols() {
                    let locked_val =
                        grid_state.locked_tiles.get([r, c]).copied().unwrap_or(1) as i32 - 1;
                    if let Some(tile_ref) = tile_3x3.get_mut(di).and_then(|row| row.get_mut(dj)) {
                        *tile_ref = locked_val;
                    }
                }
            }
        }

        let target_row = 2 - i;
        let target_col = 2 - j;
        let pattern_key = PatternKey::new(&tile_3x3, target_row, target_col);

        let compatible_bitset = cache.get_or_compute_pattern(pattern_key, || {
            find_compatible_values_at_offset_bitset(
                &tile_3x3,
                source_tiles,
                &step_data.tile_compatibility_rules,
                step_data.unique_cell_count,
                target_row,
                target_col,
            )
        });

        result_bitset = match result_bitset {
            None => Some(compatible_bitset.clone()),
            Some(current) => {
                let intersection = current.intersection(compatible_bitset);

                if intersection.is_empty() {
                    return vec![];
                }

                Some(intersection)
            }
        };
    }

    result_bitset
        .unwrap_or_else(|| TileBitset::new(step_data.unique_cell_count))
        .to_vec()
}

/// Match tile pattern against source tiles and return compatible center values
fn find_compatible_values_at_offset_bitset(
    tile_pattern: &[[i32; 3]; 3],
    source_tiles: &[Tile],
    dispatch_rules: &HashMap<Vec<u8>, Vec<usize>>,
    unique_cell_count: usize,
    target_row: usize,
    target_col: usize,
) -> TileBitset {
    let tile_booleans = convert_tile_to_membership_booleans(tile_pattern, unique_cell_count);
    let potential_sources = dispatch_rules
        .get(&tile_booleans)
        .cloned()
        .unwrap_or_default();

    let mut result = TileBitset::new(unique_cell_count);

    // Pattern uses -1 as wildcard to match any value
    let tile_pattern: [[i32; 3]; 3] =
        tile_pattern.map(|row| row.map(|val| if val == 0 { -1 } else { val }));

    for &ref_index in &potential_sources {
        if ref_index > 0 {
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

            if matches
                && let Some(row) = source_tile.get(target_row)
                    && let Some(&val) = row.get(target_col) {
                        result.insert(val);
                    }
        }
    }

    result
}

/// Extract probability values for all tile types at the specified position
pub fn get_tile_probabilities_at_position(
    grid_state: &GridState,
    position: [i32; 2],
    system_offset: [i32; 2],
) -> Vec<f64> {
    let row = (position[0] + system_offset[0]) as usize;
    let col = (position[1] + system_offset[1]) as usize;

    let mut probabilities = Vec::with_capacity(grid_state.unique_cell_count);
    for i in 0..grid_state.unique_cell_count {
        let prob = grid_state
            .tile_probabilities
            .get(i)
            .and_then(|probs| probs.get([row, col]))
            .copied()
            .unwrap_or(0.0);
        probabilities.push(prob);
    }

    probabilities
}

/// Apply density correction to maintain source distribution ratios
///
/// Uses error function-based correction to counteract deviation from
/// expected tile ratios during stochastic selection. Works in log space.
pub fn density_corrected_log_tile_weights(
    viable_tiles: &[usize],
    all_probabilities: &[f64],
    selection_tally: &[usize],
    source_ratios: &[f64],
    total_placed: usize,
    deviations: &[f64],
) -> Vec<f64> {
    let correction = optimal_density_correction(
        all_probabilities,
        selection_tally,
        source_ratios,
        total_placed,
        deviations,
    );

    let mut viable_log_corrected = Vec::with_capacity(viable_tiles.len());
    for &tile_ref in viable_tiles {
        let prob = all_probabilities.get(tile_ref - 1).copied().unwrap_or(0.0);
        let log_prob = prob.ln();
        let correction_val = correction.get(tile_ref - 1).copied().unwrap_or(0.0);
        viable_log_corrected.push(log_prob + correction_val);
    }

    let mean_log_prob =
        viable_log_corrected.iter().sum::<f64>() / viable_log_corrected.len() as f64;

    viable_log_corrected
        .iter()
        .map(|&log_prob| (log_prob - mean_log_prob))
        .collect()
}

/// Calculate correction coefficients based on current and projected deviations
///
/// Correction strength adapts based on overall deviation magnitude,
/// targeting gradual convergence to source distribution
pub fn optimal_density_correction(
    probabilities: &[f64],
    present_tally: &[usize],
    source_ratios: &[f64],
    total_placed: usize,
    deviations: &[f64],
) -> Vec<f64> {
    let deviation: f64 = source_ratios
        .iter()
        .zip(deviations)
        .map(|(ratio, dev)| ratio * dev.abs())
        .sum();

    let density_correction_threshold = 0.10;
    let density_correction_steepness = 0.05;
    let density_minimum_strength = 0.10;

    let correction_strength = 1.0
        / (1.0
            + (-density_correction_steepness
                * (deviation.mul_add(200.0, -density_correction_threshold)))
            .exp());
    let correction_strength = correction_strength.max(density_minimum_strength);

    let projected_deviation = calculate_projected_deviation(
        source_ratios,
        present_tally,
        probabilities,
        deviations,
        total_placed,
    );

    let deviation_derivative = calculate_deviation_derivative(
        source_ratios,
        present_tally,
        probabilities,
        deviations,
        total_placed,
    );

    let density_improvement_target = 0.05_f64;
    let target_deviation =
        projected_deviation * density_improvement_target.mul_add(-correction_strength, 1.0);

    let scale = (target_deviation - projected_deviation) / deviation_derivative;

    deviations
        .iter()
        .map(|&dev| scale * (-dev * dev.abs()))
        .collect()
}

/// Project future deviation after placing the next tile
///
/// The correct distribution here would be a binomial, I've use the approximating normal for speed.
pub fn calculate_projected_deviation(
    source_ratios: &[f64],
    present_tally: &[usize],
    probabilities: &[f64],
    deviations: &[f64],
    total_placed: usize,
) -> f64 {
    source_ratios
        .iter()
        .zip(present_tally)
        .zip(probabilities)
        .zip(deviations)
        .map(|(((ratio, &placed), &prob), &dev)| {
            let placed_f64 = placed as f64;
            let total_f64 = total_placed as f64;

            let arg = (-placed_f64 - prob + ratio + ratio * total_f64)
                / (2.0_f64.sqrt() * ((1.0 - ratio) * ratio * (1.0 + total_f64)).sqrt());

            -0.5 * erf(arg) * dev.signum()
        })
        .sum()
}

/// Calculate sensitivity of deviation to probability changes
///
/// Used to scale correction coefficients for stable convergence
pub fn calculate_deviation_derivative(
    source_ratios: &[f64],
    present_tally: &[usize],
    probabilities: &[f64],
    deviations: &[f64],
    total_placed: usize,
) -> f64 {
    let total_f64 = total_placed as f64;

    source_ratios
        .iter()
        .zip(present_tally)
        .zip(probabilities)
        .zip(deviations)
        .map(|(((&ratio, &placed), &prob), &dev)| {
            let placed_f64 = placed as f64;

            let numerator = ratio.mul_add(-(1.0 + total_f64), placed_f64 + prob).powi(2);
            let denominator = 2.0 * (1.0 - ratio) * ratio * (1.0 + total_f64);

            let exp_term = (-numerator / denominator).exp();
            let sqrt_term = (std::f64::consts::PI * denominator.abs()).sqrt();

            let derivative = exp_term * prob * ratio / sqrt_term;
            (-dev * dev.abs()) * derivative * dev.signum()
        })
        .sum()
}

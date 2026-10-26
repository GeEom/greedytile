use crate::math::convert::{usize_to_f64, usize_to_i32};
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
use std::collections::BTreeMap;

/// How many of the best-connected cells go on to the final draw
pub const ADJACENCY_CANDIDATES_CONSIDERED: usize = 20;
/// How many cells the final draw picks from
pub const CANDIDATES_CONSIDERED: usize = 15;
/// Power on the distance probabilities in the tile draw. Neighbours' influences
/// are multiplied as if independent; calibrated on the examples' pair statistics.
pub const DISTANCE_WEIGHT_STRENGTH: f64 = 2.0;

/// The tiles that fit at `position`. Each of its nine neighbourhoods can rule
/// some out, and we stop as soon as none are left.
pub fn compute_viable_tiles_at_position(
    grid_state: &GridState,
    position: [i32; 2],
    system_offset: [i32; 2],
    source_tiles: &[Tile],
    step_data: &StepData,
    cache: &mut ViableTilesCache,
) -> Vec<usize> {
    // The centre constrains most, so it goes first
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
            system_offset,
            [
                position[0] + usize_to_i32(i) - 1,
                position[1] + usize_to_i32(j) - 1,
            ],
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
                    let locked_val = grid_state
                        .locked_tiles
                        .get([r, c])
                        .copied()
                        .unwrap_or(1)
                        .cast_signed()
                        - 1;
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

/// The values source tiles matching `tile_pattern` have at the target cell
fn find_compatible_values_at_offset_bitset(
    tile_pattern: &[[i32; 3]; 3],
    source_tiles: &[Tile],
    dispatch_rules: &BTreeMap<Vec<u8>, Vec<usize>>,
    unique_cell_count: usize,
    target_row: usize,
    target_col: usize,
) -> TileBitset {
    let tile_booleans = convert_tile_to_membership_booleans(tile_pattern, unique_cell_count);
    let potential_sources = dispatch_rules
        .get(&tile_booleans)
        .map_or(&[][..], Vec::as_slice);

    let mut result = TileBitset::new(unique_cell_count);

    // Pattern uses -1 as wildcard to match any value
    let tile_pattern: [[i32; 3]; 3] =
        tile_pattern.map(|row| row.map(|val| if val == 0 { -1 } else { val }));

    for &ref_index in potential_sources {
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
                && let Some(&val) = row.get(target_col)
            {
                result.insert(val);
            }
        }
    }

    result
}

/// Each tile type's probability at `position`, normalised to sum to one
#[must_use]
pub fn get_tile_probabilities_at_position(
    grid_state: &GridState,
    position: [i32; 2],
    system_offset: [i32; 2],
) -> Vec<f64> {
    // Left of or above the grid, every lookup misses
    let row = usize::try_from(position[0] + system_offset[0]).unwrap_or(usize::MAX);
    let col = usize::try_from(position[1] + system_offset[1]).unwrap_or(usize::MAX);

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

    // Stored values are unnormalised products of every nearby influence. Scale by
    // the largest first so the sum cannot overflow.
    let max_prob = probabilities.iter().copied().fold(0.0, f64::max);
    if !(max_prob.is_finite() && max_prob > 0.0) {
        let uniform = 1.0 / usize_to_f64(probabilities.len().max(1));
        return vec![uniform; probabilities.len()];
    }
    let total = probabilities.iter().map(|p| p / max_prob).sum::<f64>();
    probabilities.iter().map(|p| p / max_prob / total).collect()
}

/// Log odds for each viable tile. They lean toward the source's colour ratios
/// and are centred on their mean.
#[must_use]
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
        let log_prob = DISTANCE_WEIGHT_STRENGTH * prob.max(f64::MIN_POSITIVE).ln();
        let correction_val = correction.get(tile_ref - 1).copied().unwrap_or(0.0);
        viable_log_corrected.push(log_prob + correction_val);
    }

    let mean_log_prob =
        viable_log_corrected.iter().sum::<f64>() / usize_to_f64(viable_log_corrected.len());

    viable_log_corrected
        .iter()
        .map(|&log_prob| log_prob - mean_log_prob)
        .collect()
}

/// How much to push each tile, more as the ratios drift further
#[must_use]
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

/// Where the drift should go after one more tile. It ought to be binomial. The
/// normal curve is close and faster.
#[must_use]
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
            let placed_f64 = usize_to_f64(placed);
            let total_f64 = usize_to_f64(total_placed);

            let arg = (-placed_f64 - prob + ratio + ratio * total_f64)
                / (2.0_f64.sqrt() * ((1.0 - ratio) * ratio * (1.0 + total_f64)).sqrt());

            -0.5 * erf(arg) * dev.signum()
        })
        .sum()
}

/// How fast that drift moves as the odds change
#[must_use]
pub fn calculate_deviation_derivative(
    source_ratios: &[f64],
    present_tally: &[usize],
    probabilities: &[f64],
    deviations: &[f64],
    total_placed: usize,
) -> f64 {
    let total_f64 = usize_to_f64(total_placed);

    source_ratios
        .iter()
        .zip(present_tally)
        .zip(probabilities)
        .zip(deviations)
        .map(|(((&ratio, &placed), &prob), &dev)| {
            let placed_f64 = usize_to_f64(placed);

            let numerator = ratio.mul_add(-(1.0 + total_f64), placed_f64 + prob).powi(2);
            let denominator = 2.0 * (1.0 - ratio) * ratio * (1.0 + total_f64);

            let exp_term = (-numerator / denominator).exp();
            let sqrt_term = (std::f64::consts::PI * denominator.abs()).sqrt();

            let derivative = exp_term * prob * ratio / sqrt_term;
            (-dev * dev.abs()) * derivative * dev.signum()
        })
        .sum()
}

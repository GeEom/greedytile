use crate::{
    algorithm::propagation::StepData,
    math::convert::usize_to_i32,
    math::probability::binomial_normal_approximate_cdf,
    spatial::{GridState, grid::BoundingBox},
};
use ndarray::Array2;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// How strongly each cell draws the next tile, and whether it may
pub struct WeightCalculationResult {
    /// How crowded each cell is by placed neighbours
    pub adjacency: Array2<f64>,
    /// The odds the draw gives each cell
    pub weights: Array2<f64>,
    /// Whether the draw may pick each cell
    pub validity: Array2<bool>,
}

/// Favour cells beside placed tiles and cells whose tiles would mend the colour
/// ratios. Uncertain or hard-to-fill cells count for less.
#[must_use]
pub fn calculate_position_selection(
    grid_state: &GridState,
    selection_tally: &[usize],
    step_data: &StepData,
    system_offset: [i32; 2],
) -> WeightCalculationResult {
    let total_placed = selection_tally.iter().sum::<usize>();

    let mut deviations = Vec::with_capacity(step_data.unique_cell_count);
    for i in 0..step_data.unique_cell_count {
        let p = step_data.source_ratios.get(i).copied().unwrap_or(0.0);
        let k = selection_tally.get(i).copied().unwrap_or(0);
        let n = total_placed;

        let cdf_value = binomial_normal_approximate_cdf(n, p, k);
        deviations.push(cdf_value - 0.5);
    }

    let max_deviation: f64 = step_data
        .source_ratios
        .iter()
        .zip(&deviations)
        .map(|(ratio, dev)| ratio * dev.abs())
        .sum::<f64>()
        * 200.0;

    let density_bias_strength = 1.0
        / (1.0
            + (-step_data.density_correction_steepness
                * (max_deviation - step_data.density_correction_threshold))
                .exp());
    let density_bias_strength = density_bias_strength.max(step_data.density_minimum_strength);

    let density_bias = density_bias_matrix(grid_state, &deviations, density_bias_strength);

    let mut adjacency_weight_matrix = Array2::<f64>::zeros((grid_state.rows(), grid_state.cols()));
    let mut validity_matrix =
        Array2::<bool>::from_elem((grid_state.rows(), grid_state.cols()), true);

    for i in 0..grid_state.rows() {
        for j in 0..grid_state.cols() {
            let adj_val = f64::from(
                grid_state
                    .adjacency_weights
                    .get([i, j])
                    .copied()
                    .unwrap_or(0)
                    .saturating_sub(1),
            );
            let locked_val = grid_state.locked_tiles.get([i, j]).copied().unwrap_or(0);

            if locked_val > 1
                && let Some(valid) = validity_matrix.get_mut([i, j])
            {
                *valid = false;
            }

            if adj_val > 0.0 {
                let weight = adj_val.powi(2);
                if let Some(adj_weight) = adjacency_weight_matrix.get_mut([i, j]) {
                    *adj_weight = weight;
                }
            }
        }
    }

    let mut weight_matrix = Array2::<f64>::zeros((grid_state.rows(), grid_state.cols()));
    for i in 0..grid_state.rows() {
        for j in 0..grid_state.cols() {
            let entropy = grid_state.entropy.get([i, j]).copied().unwrap_or(0.0);
            let feasibility = grid_state.feasibility.get([i, j]).copied().unwrap_or(0.0);

            if entropy > 0.0 && feasibility > 0.0 {
                let adj_weight = adjacency_weight_matrix.get([i, j]).copied().unwrap_or(0.0);
                let dens_bias = density_bias.get([i, j]).copied().unwrap_or(1.0);
                if let Some(weight) = weight_matrix.get_mut([i, j]) {
                    *weight = adj_weight * dens_bias / (feasibility * entropy);
                }
            }
        }
    }

    if let Some(bounds) = &grid_state.generation_bounds {
        apply_boundary_mask(&mut validity_matrix, bounds, system_offset);
    }

    WeightCalculationResult {
        adjacency: adjacency_weight_matrix,
        weights: weight_matrix,
        validity: validity_matrix,
    }
}

/// How far a tile at each cell would pull the colour ratios back to the source
fn density_bias_matrix(
    grid_state: &GridState,
    deviations: &[f64],
    density_bias_strength: f64,
) -> Array2<f64> {
    let mut density_bias = Array2::<f64>::ones((grid_state.rows(), grid_state.cols()));

    for i in 0..grid_state.rows() {
        for j in 0..grid_state.cols() {
            let mut dot_product = 0.0;
            let mut cell_total = 0.0;
            for (k, deviation) in deviations
                .iter()
                .enumerate()
                .take(grid_state.unique_cell_count)
            {
                let sign_dev = if deviation >= &0.0 { 1.0 } else { -1.0 };
                let exp_abs_dev = deviation.abs().exp();
                let matrix_val = grid_state
                    .tile_probabilities
                    .get(k)
                    .and_then(|probs| probs.get([i, j]))
                    .copied()
                    .unwrap_or(0.0);
                let term = sign_dev * exp_abs_dev * matrix_val;
                dot_product += term;
                cell_total += matrix_val;
            }
            // The stored probabilities are raw products and need dividing by their total
            if cell_total.is_finite() && cell_total > 0.0 {
                dot_product /= cell_total;
            }
            if let Some(bias) = density_bias.get_mut([i, j]) {
                *bias = density_bias_strength.mul_add(dot_product.exp(), 1.0);
            }
        }
    }

    density_bias
}

/// Cells outside the generation bounds can never be picked
fn apply_boundary_mask(
    validity_matrix: &mut Array2<bool>,
    bounds: &BoundingBox,
    system_offset: [i32; 2],
) {
    for i in 0..validity_matrix.nrows() {
        for j in 0..validity_matrix.ncols() {
            let world_pos = [
                usize_to_i32(i) - system_offset[0],
                usize_to_i32(j) - system_offset[1],
            ];

            if !bounds.contains(world_pos) {
                validity_matrix[[i, j]] = false;
            }
        }
    }
}

#[derive(Clone)]
struct IndexValue {
    index: [usize; 2],
    value: f64,
}

impl PartialEq for IndexValue {
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&other.value)
    }
}

impl Eq for IndexValue {}

impl Ord for IndexValue {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value
            .partial_cmp(&other.value)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for IndexValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Returns top K 2D indices with highest values, filtered by validity matrix
#[must_use]
pub fn top_k_valid_indices(
    matrix: &Array2<f64>,
    validity: &Array2<bool>,
    k: usize,
) -> Vec<[usize; 2]> {
    let (rows, cols) = matrix.dim();

    // Min-heap allows O(nlogk) selection of top k from n elements
    let mut heap = BinaryHeap::with_capacity(k + 1);

    for i in 0..rows {
        for j in 0..cols {
            // Only consider valid positions
            if !validity[[i, j]] {
                continue;
            }

            let value = matrix[[i, j]];

            if heap.len() < k {
                heap.push(Reverse(IndexValue {
                    index: [i, j],
                    value,
                }));
            } else if let Some(Reverse(min_elem)) = heap.peek()
                && value > min_elem.value
            {
                heap.pop();
                heap.push(Reverse(IndexValue {
                    index: [i, j],
                    value,
                }));
            }
        }
    }

    heap.into_iter().map(|Reverse(iv)| iv.index).collect()
}

/// Returns top K indices from a given set of indices based on their matrix values
#[must_use]
pub fn top_k_from_indices(
    matrix: &Array2<f64>,
    indices: &[[usize; 2]],
    k: usize,
) -> Vec<[usize; 2]> {
    let mut heap = BinaryHeap::with_capacity(k + 1);

    for &[i, j] in indices {
        let value = matrix[[i, j]];

        if heap.len() < k {
            heap.push(Reverse(IndexValue {
                index: [i, j],
                value,
            }));
        } else if let Some(Reverse(min_elem)) = heap.peek()
            && value > min_elem.value
        {
            heap.pop();
            heap.push(Reverse(IndexValue {
                index: [i, j],
                value,
            }));
        }
    }

    heap.into_iter().map(|Reverse(iv)| iv.index).collect()
}

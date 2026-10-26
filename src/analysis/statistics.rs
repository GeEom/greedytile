//! Pattern statistics preprocessing using kernel density estimation and spatial analysis

use crate::io::error::AlgorithmError;
use crate::math::convert::{u64_to_f64, usize_to_f64};
use crate::math::interpolation::Cubic;
use crate::math::probability::erf;
use ndarray::{Array2, Array4};
use std::collections::BTreeMap;

type TaperedInterpolationFn = Box<dyn Fn(f64) -> f64>;

/// Distance-frequency pair for spatial relationship analysis
#[derive(Debug, Clone)]
pub struct DistanceFrequency {
    /// Euclidean distance between tile positions
    pub distance: f64,
    /// Number of occurrences at this distance
    pub frequency: usize,
}

/// Aggregated distance statistics for a specific tile value pair
#[derive(Debug, Clone)]
pub struct IntegerPairDistances {
    /// Source tile value
    pub from_value: usize,
    /// Target tile value
    pub to_value: usize,
    /// All observed distances and their frequencies
    pub distances: Vec<DistanceFrequency>,
}

/// Kernel density estimator for tile pair spatial relationships
#[derive(Debug, Clone)]
pub struct SmoothKernelDistribution {
    /// Source and target tile value pair
    pub pair: (usize, usize),
    /// Distance values from the source data
    pub data_points: Vec<f64>,
    /// Frequency weights for each data point
    pub weights: Vec<f64>,
    /// Gaussian kernel bandwidth parameter
    pub bandwidth: f64,
}

impl SmoothKernelDistribution {
    /// Create a new kernel density estimator from weighted distance data
    #[must_use]
    pub fn new(pair: (usize, usize), weighted_data: Vec<(f64, f64)>) -> Self {
        let (data_points, weights): (Vec<f64>, Vec<f64>) = weighted_data.into_iter().unzip();

        Self {
            pair,
            data_points,
            weights,
            bandwidth: 1.0,
        }
    }

    /// Calculate PDF at point x using reflection at x=0 boundary to handle edge effects
    #[must_use]
    pub fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }

        let h = self.bandwidth;
        let total_weight = self.weights.iter().sum::<f64>();

        let mut sum = 0.0;
        for (x_i, w_i) in self.data_points.iter().zip(self.weights.iter()) {
            let u = (x - x_i) / h;
            let gaussian = (-0.5 * u.powi(2)).exp() / (2.0 * std::f64::consts::PI).sqrt();

            let u_reflected = (x + x_i) / h;
            let gaussian_reflected =
                (-0.5 * u_reflected.powi(2)).exp() / (2.0 * std::f64::consts::PI).sqrt();

            sum += w_i * (gaussian + gaussian_reflected);
        }

        sum / (total_weight * h)
    }
}

/// Preprocesses source pattern statistics into probability influence matrices
pub struct Processor {
    /// Source tile grid data
    source_data: Array2<usize>,
    /// Frequency ratios for each tile type in the source
    source_ratios: Vec<f64>,
    /// Maximum distance for pattern influence effects
    pattern_influence_distance: usize,
    /// Radius for grid extension operations
    grid_extension_radius: usize,
}

impl Processor {
    /// Create a new processor with source data and configuration parameters
    #[must_use]
    pub const fn new(
        source_data: Array2<usize>,
        source_ratios: Vec<f64>,
        pattern_influence_distance: usize,
        grid_extension_radius: usize,
    ) -> Self {
        Self {
            source_data,
            source_ratios,
            pattern_influence_distance,
            grid_extension_radius,
        }
    }

    /// Extract all pairwise tile distances from the source pattern
    #[must_use]
    pub fn calculate_integer_pair_distances(&self) -> Vec<IntegerPairDistances> {
        let (rows, cols) = self.source_data.dim();

        let mut coordinates_by_value: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();

        for i in 0..rows {
            for j in 0..cols {
                let value = self.source_data.get([i, j]).copied().unwrap_or(0);
                coordinates_by_value.entry(value).or_default().push((i, j));
            }
        }

        let mut distance_groups: BTreeMap<(usize, usize), Vec<u64>> = BTreeMap::new();

        for (&value1, coords1) in &coordinates_by_value {
            for (&value2, coords2) in &coordinates_by_value {
                let mut squared_distances = Vec::new();

                for &(i1, j1) in coords1 {
                    for &(i2, j2) in coords2 {
                        if (i1, j1) != (i2, j2) {
                            let di = i1.abs_diff(i2);
                            let dj = j1.abs_diff(j2);
                            let squared_distance = (di * di + dj * dj) as u64;
                            squared_distances.push(squared_distance);
                        }
                    }
                }

                if !squared_distances.is_empty() {
                    distance_groups.insert((value1, value2), squared_distances);
                }
            }
        }

        let mut result = Vec::new();

        for ((from_value, to_value), squared_distances) in distance_groups {
            let mut squared_distance_counts: BTreeMap<u64, usize> = BTreeMap::new();

            for squared_distance in squared_distances {
                *squared_distance_counts.entry(squared_distance).or_insert(0) += 1;
            }

            // Defer sqrt computation until after grouping for efficiency
            let mut distance_frequencies: Vec<DistanceFrequency> = squared_distance_counts
                .into_iter()
                .map(|(squared_distance, frequency)| DistanceFrequency {
                    distance: u64_to_f64(squared_distance).sqrt(),
                    frequency,
                })
                .collect();

            distance_frequencies.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            result.push(IntegerPairDistances {
                from_value,
                to_value,
                distances: distance_frequencies,
            });
        }

        result.sort_by_key(|item| (item.from_value, item.to_value));

        result
    }

    /// Convert distance statistics into smooth kernel density distributions
    #[must_use]
    pub fn create_smooth_kernel_distributions(
        pair_distances: &[IntegerPairDistances],
    ) -> Vec<SmoothKernelDistribution> {
        let mut distributions = Vec::new();

        for pair_data in pair_distances {
            if pair_data.distances.is_empty() {
                continue;
            }

            let weighted_data: Vec<(f64, f64)> = pair_data
                .distances
                .iter()
                .map(|df| (df.distance, usize_to_f64(df.frequency)))
                .collect();

            let dist = SmoothKernelDistribution::new(
                (pair_data.from_value, pair_data.to_value),
                weighted_data,
            );

            distributions.push(dist);
        }

        distributions
    }

    /// Fit, per source value, a spline of each pair's log density relative to their mixture
    /// # Errors
    /// Fails when the sample points cannot be fitted.
    pub fn create_density_interpolations(
        distributions: &[SmoothKernelDistribution],
        exponential_sample_points: &[f64],
    ) -> Result<Vec<Vec<Cubic>>, AlgorithmError> {
        let total_distributions = usize_to_f64(distributions.len());

        let mut grouped: BTreeMap<usize, Vec<&SmoothKernelDistribution>> = BTreeMap::new();
        for dist in distributions {
            grouped.entry(dist.pair.0).or_default().push(dist);
        }

        let mut sorted_groups: Vec<(usize, Vec<&SmoothKernelDistribution>)> =
            grouped.into_iter().collect();
        sorted_groups.sort_by_key(|(k, _)| *k);

        let mut all_interpolations = Vec::new();

        for (_source_value, group) in sorted_groups {
            let mut group_interpolations = Vec::new();

            for dist in &group {
                let mut x_values = Vec::new();
                let mut y_values = Vec::new();

                for &x in exponential_sample_points {
                    let pdf_single = dist.pdf(x);

                    let pdf_mixture: f64 =
                        group.iter().map(|d| d.pdf(x)).sum::<f64>() / usize_to_f64(group.len());

                    let n = usize_to_f64(group.len());
                    let ratio = if pdf_mixture > 0.0 && pdf_single > 0.0 {
                        (pdf_single * n / pdf_mixture).ln() - total_distributions.ln()
                    } else {
                        0.0 - total_distributions.ln()
                    };

                    x_values.push(x);
                    y_values.push(ratio);
                }

                let interpolation =
                    Cubic::new(x_values, y_values).map_err(|e| AlgorithmError::Computation {
                        operation: "cubic interpolation",
                        reason: e.to_string(),
                    })?;
                group_interpolations.push(interpolation);
            }

            all_interpolations.push(group_interpolations);
        }

        Ok(all_interpolations)
    }

    /// Blend each fit into the plain source ratio by the pattern influence distance
    fn apply_locality_tapering(
        &self,
        density_interpolations: &[Vec<Cubic>],
    ) -> Vec<Vec<TaperedInterpolationFn>> {
        let pattern_influence_distance = usize_to_f64(self.pattern_influence_distance);
        let log_ratios: Vec<f64> = self.source_ratios.iter().map(|r| r.ln()).collect();

        let mut tapered_interpolations: Vec<Vec<TaperedInterpolationFn>> = Vec::new();

        for interpolations in density_interpolations {
            let mut tapered_group: Vec<TaperedInterpolationFn> = Vec::new();

            for (target_index, interpolation) in interpolations.iter().enumerate() {
                let interp_clone = interpolation.clone();
                let log_ratio = log_ratios.get(target_index).copied().unwrap_or(0.0);
                let source_min = pattern_influence_distance;

                let tapered_fn = move |x: f64| -> f64 {
                    if x <= source_min {
                        let taper = Self::locality_taper_static(x, source_min);
                        let source_val = interp_clone.evaluate(x).unwrap_or(0.0);
                        source_val.mul_add(1.0 - taper, taper * log_ratio)
                    } else {
                        log_ratio
                    }
                };

                tapered_group.push(Box::new(tapered_fn));
            }

            tapered_interpolations.push(tapered_group);
        }

        tapered_interpolations
    }

    fn locality_taper_static(x: f64, k: f64) -> f64 {
        let sqrt_k = k.sqrt();
        let erf_max = erf(sqrt_k / 2.0);
        let erf_val = erf(sqrt_k / 2.0 - x / sqrt_k);

        0.5 - erf_val / (2.0 * erf_max)
    }

    /// Element [from][to][di][dj] is how a `from` tile changes the odds of `to` at (di, dj)
    fn compute_probability_influence_matrices(
        &self,
        tapered_interpolations: &[Vec<TaperedInterpolationFn>],
    ) -> crate::io::error::Result<Array4<f64>> {
        let unique_cell_count = self.source_ratios.len();
        let matrix_size = 2 * self.grid_extension_radius + 1;
        let radius = i32::try_from(self.grid_extension_radius).map_err(|_e| {
            crate::io::error::computation_error(
                "matrix computation",
                &format!(
                    "grid extension radius {} too large for i32",
                    self.grid_extension_radius
                ),
            )
        })?;

        let mut distance_matrix = Array2::<f64>::zeros((matrix_size, matrix_size));
        for i in 0..matrix_size {
            for j in 0..matrix_size {
                let di = i32::try_from(i).map_err(|_e| {
                    crate::io::error::computation_error(
                        "matrix computation",
                        &format!("matrix index {i} too large for i32"),
                    )
                })? - radius;
                let dj = i32::try_from(j).map_err(|_e| {
                    crate::io::error::computation_error(
                        "matrix computation",
                        &format!("matrix index {j} too large for i32"),
                    )
                })? - radius;
                let dist = f64::from(di * di + dj * dj).sqrt();
                if let Some(dist_val) = distance_matrix.get_mut([i, j]) {
                    *dist_val = 1.0 / dist.max(1.0);
                }
            }
        }

        let mut probability_influence_matrices = Array4::<f64>::zeros((
            unique_cell_count,
            unique_cell_count,
            matrix_size,
            matrix_size,
        ));

        for (from_index, group) in tapered_interpolations.iter().enumerate() {
            for (to_index, interpolation) in group.iter().enumerate() {
                for i in 0..matrix_size {
                    for j in 0..matrix_size {
                        let di = (i32::try_from(i).map_err(|_e| {
                            crate::io::error::computation_error(
                                "matrix computation",
                                &format!("matrix index {i} too large for i32"),
                            )
                        })? - radius)
                            .abs();
                        let dj = (i32::try_from(j).map_err(|_e| {
                            crate::io::error::computation_error(
                                "matrix computation",
                                &format!("matrix index {j} too large for i32"),
                            )
                        })? - radius)
                            .abs();
                        let dist = f64::from(di * di + dj * dj).sqrt();

                        let interp_val = interpolation(dist);
                        let dist_val = distance_matrix.get([i, j]).copied().unwrap_or(1.0);
                        if let Some(prob_val) =
                            probability_influence_matrices.get_mut([from_index, to_index, i, j])
                        {
                            *prob_val = dist_val * interp_val.exp();
                        }
                    }
                }
            }
        }

        Ok(probability_influence_matrices)
    }

    /// Turn the source's pair distances into probability influence matrices
    /// # Errors
    /// Fails when a fit or the matrix computation fails.
    pub fn preprocess_pattern_statistics(
        &self,
        exponential_sample_points: &[f64],
    ) -> crate::io::error::Result<Array4<f64>> {
        let pair_distances = self.calculate_integer_pair_distances();
        let distributions = Self::create_smooth_kernel_distributions(&pair_distances);
        let density_interpolations =
            Self::create_density_interpolations(&distributions, exponential_sample_points)?;
        let tapered_interpolations = self.apply_locality_tapering(&density_interpolations);
        let probability_influence_matrices =
            self.compute_probability_influence_matrices(&tapered_interpolations)?;
        Ok(probability_influence_matrices)
    }
}

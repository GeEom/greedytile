use crate::{
    algorithm::cache::ViableTilesCache,
    algorithm::feasibility::FeasibilityCountLayer,
    algorithm::propagation::StepData,
    algorithm::propagation::{
        ForcedPipeline, check_for_contradiction, detect_forced_positions,
        update_feasibility_counts, update_grid_state, update_probabilities_and_entropy,
    },
    algorithm::selection::{
        ADJACENCY_CANDIDATES_CONSIDERED, CANDIDATES_CONSIDERED, compute_viable_tiles_at_position,
        density_corrected_log_tile_weights, get_tile_probabilities_at_position,
    },
    analysis::patterns::ImageProcessor,
    analysis::statistics::Processor,
    analysis::weights::{calculate_position_selection, top_k_from_indices, top_k_valid_indices},
    io::analysis::AnalysisCapture,
    io::prefill::{PrefillData, PrefillPlacement},
    io::visualization::VisualizationCapture,
    math::convert::{usize_to_f64, usize_to_i32},
    math::probability::binomial_normal_approximate_cdf,
    spatial::GridState,
    spatial::tiles::TileExtractor,
};
use ndarray::Array4;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::io::Write;

/// Algorithm parameters controlling pattern extraction and selection behavior
#[derive(Clone, Copy, Debug)]
pub struct AlgorithmConfig {
    /// Number of top candidates to consider for selection
    pub candidates_considered: usize,
    /// Number of top adjacency candidates to consider
    pub adjacency_candidates_considered: usize,
    /// Maximum distance for pattern influence effects
    pub pattern_influence_distance: usize,
    /// Radius for grid extension operations
    pub grid_extension_radius: usize,
    /// Size of extracted tile patterns (must be odd)
    pub tile_size: usize,
    /// Whether to include rotated versions of tiles
    pub include_rotations: bool,
    /// Whether to include reflected versions of tiles
    pub include_reflections: bool,
    /// Optional generation bounds (width, height)
    pub bounds: Option<(usize, usize)>,
}

/// Algorithm state built from the bundled source image
pub struct InitialData {
    /// Algorithm parameters and source pattern data
    pub step_data: StepData,
    /// Grid holding the first placement's surroundings
    pub grid_state: GridState,
    /// Offset for coordinate system transformations
    pub system_offset: [i32; 2],
    /// 4D probability influence matrices for pattern matching
    pub probability_influence_matrices: Array4<f64>,
    /// Tile reference chosen for the first placement
    pub selected_cell_reference: usize,
    /// Coordinates of the first placement
    pub selection_coordinates: [i32; 2],
    /// Count of each tile type selected, all zero
    pub selection_tally: Vec<usize>,
    /// RGBA color for each tile type
    pub color_mapping: Vec<[u8; 4]>,
}

/// Build the starting state from `data/a.png`
/// # Errors
/// Fails when the image cannot load or its statistics cannot be computed.
pub fn load_and_initialize_data(
    seed: u64,
    include_rotations: bool,
    include_reflections: bool,
) -> crate::io::error::Result<InitialData> {
    let image_processor = ImageProcessor::from_png_file("data/a.png")?;

    let source_ratios = image_processor.source_ratios().to_vec();
    let unique_cell_count = image_processor.unique_cell_count();
    let extension_radius = image_processor.grid_extension_radius();
    let grid_extension_radius = usize_to_i32(extension_radius);
    let pattern_influence_distance = image_processor.pattern_influence_distance();
    let color_mapping = image_processor.color_mapping().to_vec();

    let source_data_2d = image_processor.source_data().clone();

    let tile_size = 3;

    let mut tile_extractor = TileExtractor::extract_tiles(
        &source_data_2d,
        tile_size,
        include_rotations,
        include_reflections,
    );
    tile_extractor.build_boolean_reference_rules(unique_cell_count);

    let source_tiles = tile_extractor.source_tiles().to_vec();
    let tile_compatibility_rules = tile_extractor.get_boolean_reference_rules().clone();

    let exponential_sample_points = TileExtractor::calculate_exponential_sample_points(
        usize_to_f64(pattern_influence_distance),
    );

    let statistics_processor = Processor::new(
        source_data_2d,
        source_ratios.clone(),
        pattern_influence_distance,
        extension_radius,
    );

    let probability_influence_matrices =
        statistics_processor.preprocess_pattern_statistics(&exponential_sample_points)?;

    let mut system_offset = [0, 0];

    // Initial tile selection weighted by source distribution
    let mut rng = StdRng::seed_from_u64(seed);
    let selected_cell_reference = select_initial_tile(&source_ratios, &mut rng);
    let selection_coordinates = [0, 0];
    let selection_tally = vec![0; unique_cell_count];
    let mut grid_state = GridState::new(1, 1, unique_cell_count);

    let (new_offset, _) =
        grid_state.extend_if_needed(system_offset, selection_coordinates, grid_extension_radius);
    system_offset = new_offset;

    let step_data = StepData {
        source_ratios,
        unique_cell_count,
        grid_extension_radius,
        density_correction_threshold: 0.10,
        density_correction_steepness: 0.05,
        density_minimum_strength: 0.10,
        source_tiles,
        tile_compatibility_rules,
    };

    Ok(InitialData {
        step_data,
        grid_state,
        system_offset,
        probability_influence_matrices,
        selected_cell_reference,
        selection_coordinates,
        selection_tally,
        color_mapping,
    })
}

/// Seeded random selector for reproducible stochastic choices
pub struct RandomSelector {
    rng: StdRng,
}

impl RandomSelector {
    /// Create a deterministic random selector
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Draw an index with probability proportional to its weight
    pub fn weighted_choice(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return 0;
        }

        let mut rand_val = self.rng.random::<f64>() * total;
        for (i, &weight) in weights.iter().enumerate() {
            rand_val -= weight;
            if rand_val <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }

    /// Draw an index by log weight, shifted by the largest so `exp` cannot overflow
    pub fn log_weighted_choice(&mut self, log_weights: &[f64]) -> usize {
        if log_weights.is_empty() {
            return 0;
        }

        let mut indices: Vec<usize> = (0..log_weights.len()).collect();
        indices.sort_by(|&a, &b| {
            log_weights
                .get(b)
                .and_then(|wb| log_weights.get(a).and_then(|wa| wb.partial_cmp(wa)))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let random_source = self.rng.random::<f64>();
        let mut cumulative = 0.0;
        let max_log_weight = indices
            .first()
            .and_then(|&idx| log_weights.get(idx))
            .copied()
            .unwrap_or(0.0);

        let shifts: Vec<f64> = indices
            .iter()
            .map(|&i| {
                log_weights
                    .get(i)
                    .map_or(0.0, |&w| (w - max_log_weight).exp())
            })
            .collect();
        let shift_sum: f64 = shifts.iter().sum::<f64>();

        for (j, &i) in indices.iter().enumerate() {
            let prob = shifts.get(j).copied().unwrap_or(0.0) / shift_sum;
            cumulative += prob;

            if cumulative >= random_source {
                return i;
            }
        }

        indices.last().copied().unwrap_or(0)
    }
}

/// Select initial tile weighted by source distribution ratios
fn select_initial_tile(source_ratios: &[f64], rng: &mut StdRng) -> usize {
    let total: f64 = source_ratios.iter().sum();
    if total <= 0.0 {
        return 1;
    }

    let mut rand_val = rng.random::<f64>() * total;
    for (i, &weight) in source_ratios.iter().enumerate() {
        rand_val -= weight;
        if rand_val <= 0.0 {
            return i + 1;
        }
    }
    source_ratios.len()
}

/// Placement decision for the current iteration
#[derive(Copy, Clone)]
struct PlacementDecision {
    /// World coordinates for placement
    world_position: [i32; 2],
    /// Tile reference (1-based index where 0=uninitialized, 1=empty, 2+=tiles)
    tile_reference: usize,
}

/// Runs generation one placement at a time, holding all of its state
pub struct GreedyStochastic {
    /// Algorithm parameters and source pattern data
    pub step_data: StepData,
    /// Current grid state with tile placements and probabilities
    pub grid_state: GridState,
    /// Offset for coordinate system transformations
    pub system_offset: [i32; 2],
    /// 4D probability influence matrices for pattern matching
    pub probability_influence_matrices: Array4<f64>,
    /// Last selected tile reference
    pub selected_cell_reference: usize,
    /// Coordinates of last selection
    pub selection_coordinates: [i32; 2],
    /// Count of each tile type selected
    pub selection_tally: Vec<usize>,
    /// Feasibility tracking layer
    pub feasibility_layer: FeasibilityCountLayer,
    /// Random number generator for stochastic selection
    pub random_selector: RandomSelector,
    /// Queue of forced position selections
    pub forced_pipeline: ForcedPipeline,
    /// Current algorithm iteration
    pub iteration: usize,
    /// RGBA color for each tile type (indexed by `tile_value` - 1)
    pub color_mapping: Vec<[u8; 4]>,
    /// Cache for viable tile computations
    pub viable_tiles_cache: ViableTilesCache,
    /// Optional visualization capture
    pub visualization: Option<VisualizationCapture>,
    /// Optional analysis metrics capture
    pub analysis: Option<AnalysisCapture>,
    /// Pre-allocated buffer to reduce allocations in hot path
    prob_buffer: Vec<f64>,
    /// Prefill data for predetermined placements
    prefill_data: Option<PrefillData>,
    /// Whether the initial placement has occurred
    initial_placement_done: bool,
}

impl GreedyStochastic {
    /// An executor for `data/a.png` with default settings
    /// # Errors
    /// Fails when the image cannot load or its statistics cannot be computed.
    pub fn new(seed: u64) -> crate::io::error::Result<Self> {
        let InitialData {
            step_data,
            grid_state,
            system_offset,
            probability_influence_matrices,
            selected_cell_reference,
            selection_coordinates,
            selection_tally,
            color_mapping,
        } = load_and_initialize_data(seed, false, false)?;

        let feasibility_layer = FeasibilityCountLayer::new(
            grid_state.rows(),
            grid_state.cols(),
            step_data.source_tiles.len(),
        );

        let random_selector = RandomSelector::new(seed);
        let forced_pipeline = ForcedPipeline::new();
        let viable_tiles_cache = ViableTilesCache::new();
        let cell_count = step_data.unique_cell_count;

        Ok(Self {
            step_data,
            grid_state,
            system_offset,
            probability_influence_matrices,
            selected_cell_reference,
            selection_coordinates,
            selection_tally,
            feasibility_layer,
            random_selector,
            forced_pipeline,
            iteration: 0,
            color_mapping,
            viable_tiles_cache,
            visualization: None,
            analysis: None,
            prob_buffer: Vec::with_capacity(cell_count),
            prefill_data: None,
            initial_placement_done: false,
        })
    }

    /// An executor for a loaded image with the given configuration
    /// # Errors
    /// Fails when the image's statistics cannot be computed.
    pub fn from_image_processor(
        image_processor: ImageProcessor,
        config: AlgorithmConfig,
        seed: u64,
    ) -> crate::io::error::Result<Self> {
        let (source_data_2d, source_ratios, unique_cell_count, _, _, color_mapping) =
            image_processor.into_parts();

        let mut tile_extractor = TileExtractor::extract_tiles(
            &source_data_2d,
            config.tile_size,
            config.include_rotations,
            config.include_reflections,
        );
        tile_extractor.build_boolean_reference_rules(unique_cell_count);

        let source_tiles = tile_extractor.source_tiles().to_vec();
        let tile_compatibility_rules = tile_extractor.get_boolean_reference_rules().clone();

        let exponential_sample_points = TileExtractor::calculate_exponential_sample_points(
            usize_to_f64(config.pattern_influence_distance),
        );

        let statistics_processor = Processor::new(
            source_data_2d,
            source_ratios.clone(),
            config.pattern_influence_distance,
            config.grid_extension_radius,
        );

        let probability_influence_matrices =
            statistics_processor.preprocess_pattern_statistics(&exponential_sample_points)?;

        let mut system_offset = [0, 0];

        // Initial tile selection weighted by source distribution
        let mut rng = StdRng::seed_from_u64(seed);
        let selected_cell_reference = select_initial_tile(&source_ratios, &mut rng);
        let selection_coordinates = [0, 0];
        let selection_tally = vec![0; unique_cell_count];
        let mut grid_state = GridState::new(1, 1, unique_cell_count);

        // Calculate generation bounds if specified
        if let Some((width, height)) = config.bounds {
            let half_width = usize_to_i32(width) / 2;
            let half_height = usize_to_i32(height) / 2;
            grid_state.generation_bounds = Some(crate::spatial::grid::BoundingBox {
                min: [-half_width, -half_height],
                max: [half_width - 1, half_height - 1],
            });
        }

        let (new_offset, _) = grid_state.extend_if_needed(
            system_offset,
            selection_coordinates,
            usize_to_i32(config.grid_extension_radius),
        );
        system_offset = new_offset;

        let step_data = StepData {
            source_ratios,
            unique_cell_count,
            grid_extension_radius: usize_to_i32(config.grid_extension_radius),
            density_correction_threshold: 0.10,
            density_correction_steepness: 0.05,
            density_minimum_strength: 0.10,
            source_tiles,
            tile_compatibility_rules,
        };

        let feasibility_layer = FeasibilityCountLayer::new(
            grid_state.rows(),
            grid_state.cols(),
            step_data.source_tiles.len(),
        );

        let random_selector = RandomSelector::new(seed);
        let forced_pipeline = ForcedPipeline::new();
        let viable_tiles_cache = ViableTilesCache::new();
        let cell_count = step_data.unique_cell_count;

        Ok(Self {
            step_data,
            grid_state,
            system_offset,
            probability_influence_matrices,
            selected_cell_reference,
            selection_coordinates,
            selection_tally,
            feasibility_layer,
            random_selector,
            forced_pipeline,
            iteration: 0,
            color_mapping,
            viable_tiles_cache,
            visualization: None,
            analysis: None,
            prob_buffer: Vec::with_capacity(cell_count),
            prefill_data: None,
            initial_placement_done: false,
        })
    }

    /// Access the current grid state
    #[must_use]
    pub const fn grid_state(&self) -> &GridState {
        &self.grid_state
    }

    /// Access tile color mapping
    #[must_use]
    pub fn color_mapping(&self) -> &[[u8; 4]] {
        &self.color_mapping
    }

    /// Queue prefill placements, growing the grid and any bounds to hold them
    /// # Errors
    /// Fails when the bounds warning cannot be written to stderr.
    pub fn apply_prefill(&mut self, prefill_data: PrefillData) -> crate::io::error::Result<()> {
        let min_coords = prefill_data.bounds.min;
        let max_coords = prefill_data.bounds.max;

        for &corner in &[
            min_coords,
            max_coords,
            [min_coords[0], max_coords[1]],
            [max_coords[0], min_coords[1]],
        ] {
            let (new_offset, _) = self.grid_state.extend_if_needed(
                self.system_offset,
                corner,
                self.step_data.grid_extension_radius,
            );
            self.system_offset = new_offset;
        }

        if let Some(gen_bounds) = &mut self.grid_state.generation_bounds
            && (!gen_bounds.contains(min_coords) || !gen_bounds.contains(max_coords))
        {
            gen_bounds.min[0] = gen_bounds.min[0].min(min_coords[0]);
            gen_bounds.min[1] = gen_bounds.min[1].min(min_coords[1]);
            gen_bounds.max[0] = gen_bounds.max[0].max(max_coords[0]);
            gen_bounds.max[1] = gen_bounds.max[1].max(max_coords[1]);

            writeln!(
                std::io::stderr().lock(),
                "Warning: Generation bounds expanded to accommodate prefill image"
            )?;
        }

        self.prefill_data = Some(prefill_data);
        Ok(())
    }

    /// Enable GIF recording of algorithm progression
    pub fn enable_visualization(&mut self, max_iterations: usize) {
        self.visualization = Some(VisualizationCapture::new(
            self.grid_state.rows(),
            self.grid_state.cols(),
            self.color_mapping.clone(),
            max_iterations,
        ));
    }

    /// Enable metrics recording for analysis
    pub fn enable_analysis(&mut self) {
        self.analysis = Some(AnalysisCapture::new(
            self.color_mapping.clone(),
            self.step_data.grid_extension_radius,
        ));
    }

    /// Export visualization as GIF if enabled
    /// # Errors
    /// Fails when visualization is off or the GIF cannot be written.
    pub fn export_visualization(&self, output_path: &str) -> crate::io::error::Result<()> {
        self.visualization.as_ref().map_or_else(
            || {
                Err(crate::io::error::AlgorithmError::InvalidParameter {
                    parameter: "visualization",
                    value: "disabled".to_string(),
                    reason: "Visualization was not enabled for this run".to_string(),
                })
            },
            |viz| viz.export_gif(output_path, crate::io::configuration::GIF_FRAME_DELAY_MS),
        )
    }

    /// Same as [`Self::run_iteration`]
    /// # Errors
    /// Fails when no position can take a tile.
    pub fn execute_iteration(&mut self) -> crate::io::error::Result<bool> {
        self.run_iteration()
    }

    /// Place one tile; returns false once the bounds are full
    /// # Errors
    /// Fails when no position can take a tile.
    pub fn run_iteration(&mut self) -> crate::io::error::Result<bool> {
        self.iteration += 1;

        if self.check_completion() {
            return Ok(false);
        }

        let decision = self.get_placement_decision()?;

        self.place_tile(decision);

        self.post_placement_updates();

        Ok(true)
    }

    /// Determine what tile to place this iteration
    fn get_placement_decision(&mut self) -> crate::io::error::Result<PlacementDecision> {
        // Special case: first iteration with no prefill
        if !self.initial_placement_done && self.prefill_data.is_none() {
            self.initial_placement_done = true;
            return Ok(PlacementDecision {
                world_position: self.selection_coordinates,
                tile_reference: self.selected_cell_reference,
            });
        }

        // Prefill comes first, then forced positions, then a weighted draw
        if let Some(prefill) = &mut self.prefill_data {
            while let Some(placement) = prefill.next_placement() {
                let [row, col] = grid_index(placement.world_position, self.system_offset);

                let is_valid = if row < self.grid_state.rows() && col < self.grid_state.cols() {
                    self.grid_state
                        .locked_tiles
                        .get([row, col])
                        .copied()
                        .unwrap_or(0)
                        <= 1
                } else {
                    true // Allow prefill to extend the grid if needed
                };

                if is_valid {
                    return Ok(PlacementDecision {
                        world_position: placement.world_position,
                        tile_reference: placement.tile_reference,
                    });
                }
            }
        }

        while let Some(forced) = self.forced_pipeline.take_next() {
            let [row, col] = grid_index(forced.coordinates, self.system_offset);

            let is_valid = if row < self.grid_state.rows() && col < self.grid_state.cols() {
                self.grid_state
                    .locked_tiles
                    .get([row, col])
                    .copied()
                    .unwrap_or(0)
                    <= 1
            } else {
                false
            };

            if is_valid {
                return Ok(PlacementDecision {
                    world_position: forced.coordinates,
                    tile_reference: forced.tile_reference,
                });
            }
        }

        self.select_random_position()
    }

    /// Select a position using the stochastic algorithm
    fn select_random_position(&mut self) -> crate::io::error::Result<PlacementDecision> {
        let weight_result = calculate_position_selection(
            &self.grid_state,
            &self.selection_tally,
            &self.step_data,
            self.system_offset,
        );

        let adjacency_candidates = top_k_valid_indices(
            &weight_result.adjacency,
            &weight_result.validity,
            ADJACENCY_CANDIDATES_CONSIDERED,
        );

        let selection_candidates = top_k_from_indices(
            &weight_result.weights,
            &adjacency_candidates,
            CANDIDATES_CONSIDERED,
        );

        let candidate_weights: Vec<f64> = selection_candidates
            .iter()
            .map(|&[i, j]| weight_result.weights.get([i, j]).copied().unwrap_or(0.0))
            .collect();

        if selection_candidates.is_empty() {
            return Err(crate::io::error::AlgorithmError::NoValidPositions {
                iteration: self.iteration,
                grid_dimensions: (self.grid_state.rows(), self.grid_state.cols()),
            });
        }

        let selected_index = self.random_selector.weighted_choice(&candidate_weights);
        let selected_pos = selection_candidates
            .get(selected_index)
            .copied()
            .unwrap_or([0, 0]);

        let world_position = [
            usize_to_i32(selected_pos[0]) - self.system_offset[0],
            usize_to_i32(selected_pos[1]) - self.system_offset[1],
        ];

        let viable_tiles = compute_viable_tiles_at_position(
            &self.grid_state,
            world_position,
            self.system_offset,
            &self.step_data.source_tiles,
            &self.step_data,
            &mut self.viable_tiles_cache,
        );

        if viable_tiles.is_empty() {
            self.resolve_deadlock(selected_pos, self.iteration);
            self.forced_pipeline = ForcedPipeline::default();

            // Retry selection after deadlock resolution
            return self.select_random_position();
        }

        let probabilities = get_tile_probabilities_at_position(
            &self.grid_state,
            world_position,
            self.system_offset,
        );

        let total_placed = self.selection_tally.iter().sum::<usize>();

        // Calculate density correction factors
        self.prob_buffer.clear();
        for i in 0..self.step_data.unique_cell_count {
            let p = self.step_data.source_ratios.get(i).copied().unwrap_or(0.0);
            let k = self.selection_tally.get(i).copied().unwrap_or(0);
            let n = total_placed;
            let cdf_value = binomial_normal_approximate_cdf(n, p, k);
            self.prob_buffer.push(cdf_value - 0.5);
        }

        let log_corrected_weights = density_corrected_log_tile_weights(
            &viable_tiles,
            &probabilities,
            &self.selection_tally,
            &self.step_data.source_ratios,
            total_placed,
            &self.prob_buffer,
        );

        let tile_idx = self
            .random_selector
            .log_weighted_choice(&log_corrected_weights);
        let tile_reference = viable_tiles.get(tile_idx).copied().unwrap_or(1);

        Ok(PlacementDecision {
            world_position,
            tile_reference,
        })
    }

    /// Place a tile and update all state
    fn place_tile(&mut self, decision: PlacementDecision) {
        if let Some(tally) = self.selection_tally.get_mut(decision.tile_reference - 1) {
            *tally += 1;
        }

        self.selected_cell_reference = decision.tile_reference;
        self.selection_coordinates = decision.world_position;

        let (new_offset, extended) = self.grid_state.extend_if_needed(
            self.system_offset,
            decision.world_position,
            self.step_data.grid_extension_radius,
        );
        self.system_offset = new_offset;

        if extended {
            self.feasibility_layer
                .extend_to(self.grid_state.rows(), self.grid_state.cols());
        }

        update_probabilities_and_entropy(
            &mut self.grid_state,
            &self.probability_influence_matrices,
            decision.tile_reference,
            decision.world_position,
            self.system_offset,
            &self.step_data,
        );

        if let Some(ref mut analysis) = self.analysis {
            analysis.record_region(
                decision.world_position[0],
                decision.world_position[1],
                &self.grid_state,
                self.system_offset,
                self.iteration,
            );
        }

        update_grid_state(
            &mut self.grid_state,
            decision.tile_reference,
            decision.world_position,
            self.system_offset,
            &mut self.visualization,
            self.iteration,
        );

        update_feasibility_counts(
            &mut self.grid_state,
            &mut self.feasibility_layer,
            decision.world_position,
            self.system_offset,
            &self.step_data,
        );
    }

    /// Perform post-placement updates
    fn post_placement_updates(&mut self) {
        let new_forced = detect_forced_positions(
            &self.grid_state,
            self.selection_coordinates,
            self.system_offset,
            &self.step_data.source_tiles,
            &self.step_data,
            &mut self.viable_tiles_cache,
        );

        self.forced_pipeline.add_positions(new_forced);

        if let Some(contradiction_pos) = check_for_contradiction(
            &self.grid_state,
            self.system_offset,
            &self.step_data,
            &mut self.viable_tiles_cache,
        ) {
            self.resolve_deadlock(contradiction_pos, self.iteration);
            self.forced_pipeline = ForcedPipeline::default();
        }
    }

    /// Check if generation is complete
    fn check_completion(&self) -> bool {
        self.grid_state
            .generation_bounds
            .as_ref()
            .is_some_and(|bounds| {
                let filled_positions = self.selection_tally.iter().sum::<usize>();
                let width = usize::try_from(bounds.max[0] - bounds.min[0] + 1).unwrap_or(0);
                let height = usize::try_from(bounds.max[1] - bounds.min[1] + 1).unwrap_or(0);
                let positions_in_bounds = width * height;

                filled_positions >= positions_in_bounds
            })
    }

    /// Unlock tiles around a contradiction to allow algorithm progression
    pub fn resolve_deadlock(&mut self, contradiction_pos: [usize; 2], iteration: usize) {
        let result = crate::algorithm::deadlock::resolve_spatial_deadlock(
            crate::algorithm::deadlock::DeadlockState {
                grid_state: &mut self.grid_state,
                feasibility_layer: &mut self.feasibility_layer,
                selection_tally: &mut self.selection_tally,
                visualization: &mut self.visualization,
            },
            contradiction_pos,
            self.system_offset,
            &self.step_data,
            &self.probability_influence_matrices,
            iteration,
        );

        // Queue replacements for any removed protected positions
        if let Some(prefill) = &mut self.prefill_data {
            for &[row, col] in &result.unlocked_positions {
                let world_pos = [
                    usize_to_i32(row) - self.system_offset[0],
                    usize_to_i32(col) - self.system_offset[1],
                ];

                if let Some(tile_ref) = prefill.is_protected(world_pos) {
                    prefill.queue_replacement(PrefillPlacement {
                        world_position: world_pos,
                        tile_reference: tile_ref,
                    });
                }
            }
        }
    }
}

/// Grid index of a world position; a position above or left of the grid
/// maps past its end, so bounds checks treat it as outside
fn grid_index(world_position: [i32; 2], system_offset: [i32; 2]) -> [usize; 2] {
    [
        usize::try_from(world_position[0] + system_offset[0]).unwrap_or(usize::MAX),
        usize::try_from(world_position[1] + system_offset[1]).unwrap_or(usize::MAX),
    ]
}

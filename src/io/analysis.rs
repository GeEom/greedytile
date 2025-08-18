//! Captures and exports algorithm metrics as animated visualization

use crate::io::error::Result;
use crate::io::visualization::VisualizationCapture;
use crate::spatial::GridState;
use crate::spatial::grid;
use std::collections::HashMap;

struct RgbaFrame {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

/// Analysis data captured at a specific grid position during algorithm execution
#[derive(Debug, Clone)]
pub struct AnalysisEvent {
    /// Absolute row coordinate
    pub row: i32,
    /// Absolute column coordinate
    pub col: i32,
    /// Algorithm iteration when this event occurred
    pub iteration: usize,
    /// Shannon entropy at this position
    pub entropy: f64,
    /// Feasibility score at this position
    pub feasibility: f64,
    /// Weighted average color based on tile probabilities
    pub weighted_color: [u8; 4],
}

/// Captures and visualizes algorithm metrics in a 2x2 grid layout:
/// - Top-left: Weighted color probabilities
/// - Top-right: Actual tile placements
/// - Bottom-left: Entropy (information content)
/// - Bottom-right: Feasibility (placement viability)
pub struct AnalysisCapture {
    events: Vec<AnalysisEvent>,
    color_mapping: Vec<[u8; 4]>,
    capture_radius: i32,
}

impl AnalysisCapture {
    /// Create a new analysis capture with color mapping and grid parameters
    pub fn new(color_mapping: Vec<[u8; 4]>, grid_extension_radius: i32) -> Self {
        Self {
            events: Vec::with_capacity(10000),
            color_mapping,
            capture_radius: grid_extension_radius,
        }
    }

    /// Calculates the weighted average color from cell probabilities
    fn calculate_weighted_color(&self, probabilities: &[f64]) -> [u8; 4] {
        let mut weighted_r = 0.0;
        let mut weighted_g = 0.0;
        let mut weighted_b = 0.0;
        let mut weighted_a = 0.0;
        let mut total_weight = 0.0;

        // Cells are 1 indexed, conversion to 0 index for the color lookup here
        for (tile_idx, &prob) in probabilities.iter().enumerate() {
            if prob > 0.0 && tile_idx > 0
                && let Some(color) = self.color_mapping.get(tile_idx - 1) {
                    weighted_r += color[0] as f64 * prob;
                    weighted_g += color[1] as f64 * prob;
                    weighted_b += color[2] as f64 * prob;
                    weighted_a += color[3] as f64 * prob;
                    total_weight += prob;
                }
        }

        if total_weight > 0.0 {
            [
                (weighted_r / total_weight).round() as u8,
                (weighted_g / total_weight).round() as u8,
                (weighted_b / total_weight).round() as u8,
                (weighted_a / total_weight).round() as u8,
            ]
        } else {
            [0, 0, 0, 255]
        }
    }

    /// Get the number of events captured
    pub const fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Records analysis data for a region around a placement
    ///
    /// Captures entropy, feasibility, and probability data within the capture radius
    pub fn record_region(
        &mut self,
        center_row: i32,
        center_col: i32,
        grid_state: &GridState,
        system_offset: [i32; 2],
        iteration: usize,
    ) {
        let (row_span, col_span) = grid::get_region_spans(
            &system_offset,
            &[center_row, center_col],
            self.capture_radius,
        );

        let row_start = row_span.start.min(grid_state.rows());
        let row_end = row_span.end.min(grid_state.rows());
        let col_start = col_span.start.min(grid_state.cols());
        let col_end = col_span.end.min(grid_state.cols());

        for row in row_start..row_end {
            for col in col_start..col_end {
                let entropy = *grid_state.entropy.get([row, col]).unwrap_or(&0.0);
                let feasibility = *grid_state.feasibility.get([row, col]).unwrap_or(&0.0);

                let mut probs = vec![0.0; grid_state.unique_cell_count + 1];
                for (i, prob_matrix) in grid_state.tile_probabilities.iter().enumerate() {
                    if let Some(prob_value) = prob_matrix.get([row, col])
                        && let Some(prob_slot) = probs.get_mut(i + 1) {
                            *prob_slot = *prob_value;
                        }
                }
                let weighted_color = self.calculate_weighted_color(&probs);

                let abs_row = row as i32 - system_offset[0];
                let abs_col = col as i32 - system_offset[1];

                self.events.push(AnalysisEvent {
                    row: abs_row,
                    col: abs_col,
                    iteration,
                    entropy,
                    feasibility,
                    weighted_color,
                });
            }
        }
    }

    /// Calculate unified bounds across all events and visualization placements
    ///
    /// Determines the minimal bounding box that contains all analysis events
    /// and tile placements to ensure consistent frame dimensions
    fn calculate_unified_bounds(
        &self,
        visualization: &VisualizationCapture,
    ) -> (i32, i32, usize, usize) {
        let mut min_row = i32::MAX;
        let mut max_row = i32::MIN;
        let mut min_col = i32::MAX;
        let mut max_col = i32::MIN;

        for event in &self.events {
            min_row = min_row.min(event.row);
            max_row = max_row.max(event.row);
            min_col = min_col.min(event.col);
            max_col = max_col.max(event.col);
        }

        let placements = visualization.get_placements();
        for placement in placements {
            min_row = min_row.min(placement.row);
            max_row = max_row.max(placement.row);
            min_col = min_col.min(placement.col);
            max_col = max_col.max(placement.col);
        }

        let total_rows = (max_row - min_row + 1) as usize;
        let total_cols = (max_col - min_col + 1) as usize;

        (min_row, min_col, total_rows, total_cols)
    }

    fn create_entropy_grid(
        &self,
        bounds: (i32, i32, usize, usize),
        up_to_iteration: usize,
    ) -> Vec<Vec<f64>> {
        let (min_row, min_col, rows, cols) = bounds;
        let mut grid = vec![vec![0.0; cols]; rows];

        for event in &self.events {
            if event.iteration <= up_to_iteration {
                let row_idx = (event.row - min_row) as usize;
                let col_idx = (event.col - min_col) as usize;
                if let Some(row) = grid.get_mut(row_idx)
                    && let Some(cell) = row.get_mut(col_idx) {
                        *cell = event.entropy;
                    }
            }
        }

        grid
    }

    fn create_feasibility_grid(
        &self,
        bounds: (i32, i32, usize, usize),
        up_to_iteration: usize,
    ) -> Vec<Vec<f64>> {
        let (min_row, min_col, rows, cols) = bounds;
        let mut grid = vec![vec![0.0; cols]; rows];

        for event in &self.events {
            if event.iteration <= up_to_iteration {
                let row_idx = (event.row - min_row) as usize;
                let col_idx = (event.col - min_col) as usize;
                if let Some(row) = grid.get_mut(row_idx)
                    && let Some(cell) = row.get_mut(col_idx) {
                        *cell = event.feasibility;
                    }
            }
        }

        grid
    }

    fn reconstruct_grid_at_iteration(
        visualization: &VisualizationCapture,
        iteration: usize,
    ) -> HashMap<(i32, i32), usize> {
        let mut grid = HashMap::new();
        for placement in visualization.get_placements() {
            if placement.iteration <= iteration {
                if let Some(tile_ref) = placement.tile_ref {
                    grid.insert((placement.row, placement.col), tile_ref as usize);
                } else {
                    grid.remove(&(placement.row, placement.col));
                }
            }
        }
        grid
    }

    fn create_weighted_color_grid(
        &self,
        bounds: (i32, i32, usize, usize),
        up_to_iteration: usize,
    ) -> Vec<Vec<[u8; 4]>> {
        let (min_row, min_col, rows, cols) = bounds;
        let mut grid = vec![vec![[0, 0, 0, 255]; cols]; rows];

        for event in &self.events {
            if event.iteration <= up_to_iteration {
                let row_idx = (event.row - min_row) as usize;
                let col_idx = (event.col - min_col) as usize;
                if let Some(row) = grid.get_mut(row_idx)
                    && let Some(cell) = row.get_mut(col_idx) {
                        *cell = event.weighted_color;
                    }
            }
        }

        grid
    }

    fn render_combined_frame(
        &self,
        visualization: &VisualizationCapture,
        bounds: (i32, i32, usize, usize),
        iteration: usize,
        _delay_ms: u32,
        max_entropy: f64,
    ) -> RgbaFrame {
        let (min_row, min_col, grid_rows, grid_cols) = bounds;

        let entropy_grid = self.create_entropy_grid(bounds, iteration);
        let feasibility_grid = self.create_feasibility_grid(bounds, iteration);
        let weighted_color_grid = self.create_weighted_color_grid(bounds, iteration);

        let tile_grid = Self::reconstruct_grid_at_iteration(visualization, iteration);

        // max_entropy ensures consistent normalization across all frames

        let padding = 2;
        let total_width = grid_cols * 2 + padding;
        let total_height = grid_rows * 2 + padding;

        let mut pixels = vec![0u8; total_width * total_height * 4];

        for row in 0..grid_rows {
            for col in 0..grid_cols {
                let pixel_idx = (row * total_width + col) * 4;
                if let Some(grid_row) = weighted_color_grid.get(row)
                    && let Some(color) = grid_row.get(col)
                        && let Some(pixel_slice) = pixels.get_mut(pixel_idx..pixel_idx + 4) {
                            pixel_slice.copy_from_slice(color);
                        }
            }
        }

        for row in 0..grid_rows {
            for col in 0..grid_cols {
                let pixel_idx = (row * total_width + col + grid_cols + padding) * 4;
                let abs_row = row as i32 + min_row;
                let abs_col = col as i32 + min_col;

                if let Some(&tile_idx) = tile_grid.get(&(abs_row, abs_col)) {
                    if tile_idx > 1
                        && let Some(color) = self.color_mapping.get(tile_idx - 2)
                            && let Some(pixel_slice) = pixels.get_mut(pixel_idx..pixel_idx + 4) {
                                pixel_slice.copy_from_slice(color);
                            }
                } else if let Some(pixel_slice) = pixels.get_mut(pixel_idx..pixel_idx + 4) {
                    pixel_slice.copy_from_slice(&[0, 0, 0, 255]);
                }
            }
        }

        for row in 0..grid_rows {
            for col in 0..grid_cols {
                let pixel_idx = ((row + grid_rows + padding) * total_width + col) * 4;
                if let Some(grid_row) = entropy_grid.get(row)
                    && let Some(&entropy) = grid_row.get(col) {
                        let normalized = if max_entropy > 0.0 {
                            (entropy / max_entropy * 255.0) as u8
                        } else {
                            0
                        };
                        if let Some(pixel_slice) = pixels.get_mut(pixel_idx..pixel_idx + 4) {
                            pixel_slice.copy_from_slice(&[normalized, normalized, normalized, 255]);
                        }
                    }
            }
        }

        for row in 0..grid_rows {
            for col in 0..grid_cols {
                let pixel_idx =
                    ((row + grid_rows + padding) * total_width + col + grid_cols + padding) * 4;
                if let Some(grid_row) = feasibility_grid.get(row)
                    && let Some(&feasibility) = grid_row.get(col) {
                        let normalized = (feasibility * 255.0) as u8;
                        if let Some(pixel_slice) = pixels.get_mut(pixel_idx..pixel_idx + 4) {
                            pixel_slice.copy_from_slice(&[normalized, normalized, normalized, 255]);
                        }
                    }
            }
        }

        let gray = [128u8, 128, 128, 255];
        for row in 0..grid_rows {
            for p in 0..padding {
                let pixel_idx = (row * total_width + grid_cols + p) * 4;
                if let Some(pixel_slice) = pixels.get_mut(pixel_idx..pixel_idx + 4) {
                    pixel_slice.copy_from_slice(&gray);
                }
                let bottom_pixel_idx =
                    ((row + grid_rows + padding) * total_width + grid_cols + p) * 4;
                if let Some(pixel_slice) = pixels.get_mut(bottom_pixel_idx..bottom_pixel_idx + 4) {
                    pixel_slice.copy_from_slice(&gray);
                }
            }
        }
        for col in 0..total_width {
            for p in 0..padding {
                let pixel_idx = ((grid_rows + p) * total_width + col) * 4;
                if let Some(pixel_slice) = pixels.get_mut(pixel_idx..pixel_idx + 4) {
                    pixel_slice.copy_from_slice(&gray);
                }
            }
        }

        RgbaFrame {
            width: total_width as u32,
            height: total_height as u32,
            pixels,
        }
    }

    /// Export analysis as animated GIF with 2x2 layout
    ///
    /// Creates an animated visualization showing the algorithm's decision-making
    /// process through entropy, feasibility, and probability metrics
    ///
    /// # Errors
    ///
    /// Returns an error if image creation or file operations fail
    pub fn export_analysis(
        &self,
        visualization: &VisualizationCapture,
        output_path: &str,
        frame_delay_ms: u32,
    ) -> Result<()> {
        use image::{Frame, RgbaImage};

        let bounds = self.calculate_unified_bounds(visualization);

        let max_iteration = self
            .events
            .iter()
            .map(|e| e.iteration)
            .max()
            .unwrap_or(0)
            .max(
                visualization
                    .get_placements()
                    .iter()
                    .map(|p| p.iteration)
                    .max()
                    .unwrap_or(0),
            );

        // Calculate global max entropy for consistent normalization
        let max_entropy = self.events.iter().map(|e| e.entropy).fold(0.0, f64::max);

        let mut frames = Vec::new();
        for iteration in 0..=max_iteration {
            let rgba_frame = self.render_combined_frame(
                visualization,
                bounds,
                iteration,
                frame_delay_ms,
                max_entropy,
            );

            let img = RgbaImage::from_raw(rgba_frame.width, rgba_frame.height, rgba_frame.pixels)
                .ok_or_else(|| crate::io::error::AlgorithmError::InvalidSourceData {
                reason: "Failed to create image from frame data".to_string(),
            })?;

            frames.push(Frame::from_parts(
                img,
                0,
                0,
                image::Delay::from_numer_denom_ms(frame_delay_ms, 1),
            ));
        }

        // Add final frame with 25x longer delay for viewing
        if !frames.is_empty() {
            let final_frame_delay = frame_delay_ms * 25;
            if let Some(last_frame_img) = frames.last().map(|f| f.buffer().clone()) {
                frames.push(Frame::from_parts(
                    last_frame_img,
                    0,
                    0,
                    image::Delay::from_numer_denom_ms(final_frame_delay, 1),
                ));
            }
        }

        if let Some(parent) = std::path::Path::new(output_path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                crate::io::error::AlgorithmError::FileSystem {
                    path: parent.to_path_buf(),
                    operation: "create directory",
                    source: e,
                }
            })?;
        }

        let file = std::fs::File::create(output_path).map_err(|e| {
            crate::io::error::AlgorithmError::FileSystem {
                path: output_path.into(),
                operation: "create file",
                source: e,
            }
        })?;

        let mut encoder = image::codecs::gif::GifEncoder::new(file);
        encoder.encode_frames(frames).map_err(|e| {
            crate::io::error::AlgorithmError::ImageExport {
                path: output_path.into(),
                source: e,
            }
        })?;

        Ok(())
    }
}

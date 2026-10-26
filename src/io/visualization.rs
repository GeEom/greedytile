//! Frame capture and GIF generation for algorithm visualization

use crate::io::error::{AlgorithmError, Result};
use crate::math::convert::usize_to_u32;
use image::{Frame, Rgba, RgbaImage};

/// Represents a single tile placement event
#[derive(Debug, Clone)]
pub struct TilePlacement {
    /// Absolute row coordinate
    pub row: i32,
    /// Absolute column coordinate  
    pub col: i32,
    /// Tile reference (None for removals)
    pub tile_ref: Option<u32>,
    /// Algorithm iteration when placed
    pub iteration: usize,
}

/// Placements and removals in the order they happened, for replay as a GIF
pub struct VisualizationCapture {
    pub(crate) placements: Vec<TilePlacement>,
    initial_dims: (usize, usize),
    color_mapping: Vec<[u8; 4]>,
    empty_color: [u8; 4],
}

impl VisualizationCapture {
    /// The average of all tile colors is used as the empty color
    #[must_use]
    pub fn new(
        initial_rows: usize,
        initial_cols: usize,
        color_mapping: Vec<[u8; 4]>,
        max_iterations: usize,
    ) -> Self {
        let empty_color = if color_mapping.is_empty() {
            [128, 128, 128, 255]
        } else {
            let mut r_sum = 0u32;
            let mut g_sum = 0u32;
            let mut b_sum = 0u32;
            let mut a_sum = 0u32;

            for color in &color_mapping {
                r_sum += u32::from(color[0]);
                g_sum += u32::from(color[1]);
                b_sum += u32::from(color[2]);
                a_sum += u32::from(color[3]);
            }

            // Each mean of u8 channels fits in a u8
            let count = usize_to_u32(color_mapping.len());
            [
                u8::try_from(r_sum / count).unwrap_or(u8::MAX),
                u8::try_from(g_sum / count).unwrap_or(u8::MAX),
                u8::try_from(b_sum / count).unwrap_or(u8::MAX),
                u8::try_from(a_sum / count).unwrap_or(u8::MAX),
            ]
        };

        Self {
            placements: Vec::with_capacity(max_iterations),
            initial_dims: (initial_rows, initial_cols),
            color_mapping,
            empty_color,
        }
    }

    /// Records a tile placement at the given position
    pub fn record_placement(&mut self, row: i32, col: i32, tile_ref: u32, iteration: usize) {
        self.placements.push(TilePlacement {
            row,
            col,
            tile_ref: Some(tile_ref),
            iteration,
        });
    }

    /// Records a tile removal at the given position
    pub fn record_removal(&mut self, row: i32, col: i32, iteration: usize) {
        self.placements.push(TilePlacement {
            row,
            col,
            tile_ref: None,
            iteration,
        });
    }

    /// Returns all recorded placement events
    #[must_use]
    pub fn get_placements(&self) -> &[TilePlacement] {
        &self.placements
    }

    /// Write the replay as a GIF, dropping frames when the delay is under what viewers show
    /// # Errors
    /// Fails when nothing was captured or the file cannot be written.
    pub fn export_gif(&self, output_path: &str, frame_delay_ms: u32) -> Result<()> {
        use crate::io::configuration::VIEWER_MIN_FRAME_DELAY_MS;

        if self.placements.is_empty() {
            return Err(AlgorithmError::InvalidSourceData {
                reason: "No tile placements captured for visualization".to_string(),
            });
        }

        let effective_delay_ms = frame_delay_ms.max(VIEWER_MIN_FRAME_DELAY_MS);
        let skip_factor = if frame_delay_ms < VIEWER_MIN_FRAME_DELAY_MS {
            VIEWER_MIN_FRAME_DELAY_MS.div_ceil(frame_delay_ms)
        } else {
            1
        };

        let (min_row, min_col, final_rows, final_cols) = self.calculate_final_bounds();

        let frames = self.generate_frames(
            min_row,
            min_col,
            final_rows,
            final_cols,
            effective_delay_ms,
            skip_factor as usize,
        )?;

        if let Some(parent) = std::path::Path::new(output_path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| AlgorithmError::FileSystem {
                path: parent.to_path_buf(),
                operation: "create directory",
                source: e,
            })?;
        }

        let file = std::fs::File::create(output_path).map_err(|e| AlgorithmError::FileSystem {
            path: output_path.into(),
            operation: "create file",
            source: e,
        })?;

        let mut encoder = image::codecs::gif::GifEncoder::new(file);
        encoder
            .encode_frames(frames)
            .map_err(|e| AlgorithmError::ImageExport {
                path: output_path.into(),
                source: e,
            })?;

        Ok(())
    }

    fn calculate_final_bounds(&self) -> (i32, i32, usize, usize) {
        if self.placements.is_empty() {
            return (0, 0, self.initial_dims.0, self.initial_dims.1);
        }

        let mut min_row = 0;
        let mut max_row = 0;
        let mut min_col = 0;
        let mut max_col = 0;

        for placement in &self.placements {
            min_row = min_row.min(placement.row);
            max_row = max_row.max(placement.row);
            min_col = min_col.min(placement.col);
            max_col = max_col.max(placement.col);
        }

        let rows = usize::try_from(max_row - min_row + 1).unwrap_or(0);
        let cols = usize::try_from(max_col - min_col + 1).unwrap_or(0);

        (min_row, min_col, rows, cols)
    }

    fn generate_frames(
        &self,
        min_row: i32,
        min_col: i32,
        rows: usize,
        cols: usize,
        delay_ms: u32,
        skip_factor: usize,
    ) -> Result<Vec<Frame>> {
        // 0 = removal, 1 = empty, 2+ = tiles
        let mut grid = vec![vec![1u32; cols]; rows];
        let mut frames = Vec::new();

        frames.push(self.render_frame(&grid, rows, cols, delay_ms)?);

        let mut frame_count = 0;

        for placement in &self.placements {
            let grid_row = usize::try_from(placement.row - min_row).unwrap_or(usize::MAX);
            let grid_col = usize::try_from(placement.col - min_col).unwrap_or(usize::MAX);

            if grid_row < rows && grid_col < cols {
                if let Some(row) = grid.get_mut(grid_row)
                    && let Some(cell) = row.get_mut(grid_col)
                {
                    *cell = placement.tile_ref.unwrap_or(0);
                }

                frame_count += 1;

                if frame_count % skip_factor == 0 {
                    frames.push(self.render_frame(&grid, rows, cols, delay_ms)?);
                }
            }
        }

        if frame_count % skip_factor != 0 {
            frames.push(self.render_frame(&grid, rows, cols, delay_ms)?);
        }

        // Final frame displays longer for better visibility
        if !frames.is_empty() {
            let final_frame_delay = delay_ms * 25;
            if let Some(last_frame_img) = frames.last().map(|f| f.buffer().clone()) {
                frames.push(Frame::from_parts(
                    last_frame_img,
                    0,
                    0,
                    image::Delay::from_numer_denom_ms(final_frame_delay, 1),
                ));
            }
        }

        Ok(frames)
    }

    fn render_frame(
        &self,
        grid: &[Vec<u32>],
        rows: usize,
        cols: usize,
        delay_ms: u32,
    ) -> Result<Frame> {
        let mut img = RgbaImage::new(usize_to_u32(cols), usize_to_u32(rows));

        for (row, row_data) in grid.iter().enumerate().take(rows) {
            for (col, &tile_ref) in row_data.iter().enumerate().take(cols) {
                let color = match tile_ref {
                    0 | 1 => Rgba(self.empty_color),
                    _ => {
                        let color_index = (tile_ref - 2) as usize;
                        let rgba =
                            self.color_mapping
                                .get(color_index)
                                .copied()
                                .ok_or_else(|| AlgorithmError::InvalidTileIndex {
                                    index: tile_ref as usize,
                                    max_tiles: self.color_mapping.len() + 1,
                                })?;
                        Rgba([rgba[0], rgba[1], rgba[2], rgba[3]])
                    }
                };

                img.put_pixel(usize_to_u32(col), usize_to_u32(row), color);
            }
        }

        Ok(Frame::from_parts(
            img,
            0,
            0,
            image::Delay::from_numer_denom_ms(delay_ms, 1),
        ))
    }

    /// Returns the total number of placement events
    #[must_use]
    pub const fn placement_count(&self) -> usize {
        self.placements.len()
    }
}

//! Image processing and pattern extraction from source images

use ndarray::{Array2, Array3};
use std::collections::HashMap;
use std::path::Path;

/// Converts images to integer-labeled grids and extracts pattern statistics
pub struct ImageProcessor {
    source_data: Array2<usize>,
    source_ratios: Vec<f64>,
    unique_cell_count: usize,
    pattern_influence_distance: usize,
    grid_extension_radius: usize,
    color_mapping: Vec<[u8; 4]>,
}

impl ImageProcessor {
    /// Load and process an image from a PNG file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file at the given path cannot be opened or read
    /// - The file is not a valid image format
    /// - The image cannot be converted to RGBA format
    pub fn from_png_file<P: AsRef<Path>>(path: P) -> crate::io::error::Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let img =
            image::open(&path_buf).map_err(|e| crate::io::error::AlgorithmError::ImageLoad {
                path: path_buf,
                source: e,
            })?;
        let rgba_img = img.to_rgba8();

        // Convert to Array3<f64> format (height, width, 4 channels)
        let (width, height) = (rgba_img.width() as usize, rgba_img.height() as usize);
        let mut image_data = Array3::zeros((height, width, 4));

        for (x, y, pixel) in rgba_img.enumerate_pixels() {
            let channels = pixel.0;
            for c in 0..4 {
                let val = channels.get(c).copied().unwrap_or(0);
                if let Some(pixel_val) = image_data.get_mut((y as usize, x as usize, c)) {
                    *pixel_val = f64::from(val) / 255.0;
                }
            }
        }

        Ok(Self::from_raw_image(&image_data))
    }

    /// Process a raw image array into integer labels
    pub fn from_raw_image(image_data: &Array3<f64>) -> Self {
        let mut color_set = std::collections::HashSet::new();
        let (height, width, _) = image_data.dim();

        for i in 0..height {
            for j in 0..width {
                let color = [
                    image_data[(i, j, 0)],
                    image_data[(i, j, 1)],
                    image_data[(i, j, 2)],
                    image_data[(i, j, 3)],
                ];
                color_set.insert(color_to_bytes(&color));
            }
        }

        // Deterministic color ordering ensures reproducible tile assignments
        let mut unique_colors_bytes: Vec<[u8; 4]> = color_set.into_iter().collect();
        unique_colors_bytes.sort_unstable();

        let mut color_mapping = HashMap::new();
        unique_colors_bytes
            .iter()
            .enumerate()
            .for_each(|(index, &color_bytes)| {
                color_mapping.insert(color_bytes, index + 1);
            });

        let mut source_data = Array2::zeros((height, width));
        for i in 0..height {
            for j in 0..width {
                let color = [
                    image_data[(i, j, 0)],
                    image_data[(i, j, 1)],
                    image_data[(i, j, 2)],
                    image_data[(i, j, 3)],
                ];
                let color_bytes = color_to_bytes(&color);
                if let Some(&mapping) = color_mapping.get(&color_bytes)
                    && let Some(data) = source_data.get_mut((i, j))
                {
                    *data = mapping;
                }
            }
        }

        let mut counts = vec![0usize; unique_colors_bytes.len()];
        for &val in &source_data {
            if val > 0
                && let Some(count) = counts.get_mut(val - 1)
            {
                *count += 1;
            }
        }

        let total: usize = counts.iter().sum();
        let source_ratios: Vec<f64> = counts
            .iter()
            .map(|&c| (c as f64) / (total as f64))
            .collect();

        let unique_cell_count = source_ratios.len();
        let pattern_influence_distance = height.min(width) / 2;
        let grid_extension_radius = pattern_influence_distance.saturating_sub(1);

        Self {
            source_data,
            source_ratios,
            unique_cell_count,
            pattern_influence_distance,
            grid_extension_radius,
            color_mapping: unique_colors_bytes,
        }
    }

    /// Load and process an image from a PNG file path
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file at the given path cannot be opened or read
    /// - The file is not a valid image format
    /// - The image cannot be converted to RGBA format
    pub fn from_png_path<P: AsRef<Path>>(path: P) -> crate::io::error::Result<Self> {
        Self::from_png_file(path)
    }

    /// Get the source pattern data grid
    pub const fn source_data(&self) -> &Array2<usize> {
        &self.source_data
    }

    /// Get the frequency ratios for each tile type
    pub fn source_ratios(&self) -> &[f64] {
        &self.source_ratios
    }

    /// Get the number of unique tile types
    pub const fn unique_cell_count(&self) -> usize {
        self.unique_cell_count
    }

    /// Returns the pattern influence distance (min dimension / 2)
    ///
    /// Used to determine how far pattern influences extend in the generated output
    pub const fn pattern_influence_distance(&self) -> usize {
        self.pattern_influence_distance
    }

    /// Returns the grid extension radius for dynamic grid growth
    pub const fn grid_extension_radius(&self) -> usize {
        self.grid_extension_radius
    }

    /// Returns RGBA values for each tile type (indexed by `tile_value` - 1)
    /// Get the RGBA color mapping for tile visualization
    pub fn color_mapping(&self) -> &[[u8; 4]] {
        &self.color_mapping
    }

    /// Consume the processor and return all its components
    pub fn into_parts(self) -> (Array2<usize>, Vec<f64>, usize, usize, usize, Vec<[u8; 4]>) {
        (
            self.source_data,
            self.source_ratios,
            self.unique_cell_count,
            self.pattern_influence_distance,
            self.grid_extension_radius,
            self.color_mapping,
        )
    }
}

fn color_to_bytes(color: &[f64; 4]) -> [u8; 4] {
    [
        (color[0] * 255.0) as u8,
        (color[1] * 255.0) as u8,
        (color[2] * 255.0) as u8,
        (color[3] * 255.0) as u8,
    ]
}

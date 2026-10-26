//! Prefill image parsing and queue management for predetermined tile placement

use crate::io::error::{AlgorithmError, Result};
use crate::spatial::grid::BoundingBox;
use std::collections::{BTreeMap, VecDeque};
use std::path::Path;

/// Single tile placement instruction
#[derive(Debug, Clone)]
pub struct PrefillPlacement {
    /// World coordinates for placement
    pub world_position: [i32; 2],
    /// Tile reference (1-based index)
    pub tile_reference: usize,
}

/// Manages prefill placements and position protection
pub struct PrefillData {
    /// Queue of positions and tiles to place
    pub placement_queue: VecDeque<PrefillPlacement>,
    /// Positions that must be maintained if removed
    pub protected_positions: BTreeMap<[i32; 2], usize>,
    /// Bounding box of all prefill positions
    pub bounds: BoundingBox,
}

impl PrefillData {
    /// Queue every pixel whose colour is in the source palette, centred on the origin
    /// # Errors
    /// Fails when the PNG cannot load or none of its colours is in the palette.
    pub fn from_png(path: &Path, color_mapping: &[[u8; 4]]) -> Result<Self> {
        let img = image::open(path).map_err(|e| AlgorithmError::ImageLoad {
            path: path.to_path_buf(),
            source: e,
        })?;

        let rgba_img = img.to_rgba8();
        let (width, height) = rgba_img.dimensions();

        // Build reverse mapping from color to tile index
        let mut color_to_tile: BTreeMap<[u8; 4], usize> = BTreeMap::new();
        for (idx, &color) in color_mapping.iter().enumerate() {
            color_to_tile.insert(color, idx + 1);
        }

        let mut placement_queue = VecDeque::new();
        let mut protected_positions = BTreeMap::new();

        let mut min_row = i32::MAX;
        let mut max_row = i32::MIN;
        let mut min_col = i32::MAX;
        let mut max_col = i32::MIN;

        // Center the prefill image at origin
        let offset_x = width.cast_signed() / 2;
        let offset_y = height.cast_signed() / 2;

        for (x, y, pixel) in rgba_img.enumerate_pixels() {
            let color = [pixel[0], pixel[1], pixel[2], pixel[3]];

            if let Some(&tile_ref) = color_to_tile.get(&color) {
                let world_x = x.cast_signed() - offset_x;
                let world_y = y.cast_signed() - offset_y;
                // Grid system expects [row, col] format, so swap x and y
                let world_pos = [world_y, world_x];

                placement_queue.push_back(PrefillPlacement {
                    world_position: world_pos,
                    tile_reference: tile_ref,
                });

                protected_positions.insert(world_pos, tile_ref);

                min_row = min_row.min(world_pos[0]);
                max_row = max_row.max(world_pos[0]);
                min_col = min_col.min(world_pos[1]);
                max_col = max_col.max(world_pos[1]);
            }
        }

        if placement_queue.is_empty() {
            return Err(AlgorithmError::InvalidSourceData {
                reason: "Prefill image contains no colors from source palette".to_string(),
            });
        }

        let bounds = BoundingBox {
            min: [min_row, min_col],
            max: [max_row, max_col],
        };

        Ok(Self {
            placement_queue,
            protected_positions,
            bounds,
        })
    }

    /// Check if a position is protected by prefill
    #[must_use]
    pub fn is_protected(&self, world_pos: [i32; 2]) -> Option<usize> {
        self.protected_positions.get(&world_pos).copied()
    }

    /// Get the next placement from the queue
    pub fn next_placement(&mut self) -> Option<PrefillPlacement> {
        self.placement_queue.pop_front()
    }

    /// Add a replacement to the front of the queue
    pub fn queue_replacement(&mut self, placement: PrefillPlacement) {
        self.placement_queue.push_front(placement);
    }
}

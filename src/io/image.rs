//! PNG export with automatic cropping and transparency handling

use crate::math::convert::usize_to_u32;
use crate::spatial::GridState;
use image::{ImageBuffer, Rgba};

#[derive(Debug)]
struct BoundingBox {
    min_row: usize,
    max_row: usize,
    min_col: usize,
    max_col: usize,
}

// Finds the minimal rectangle containing all non-empty tiles
fn calculate_bounding_box(grid_state: &GridState) -> Option<BoundingBox> {
    let mut min_row = usize::MAX;
    let mut max_row = 0;
    let mut min_col = usize::MAX;
    let mut max_col = 0;
    let mut found_tiles = false;

    for row in 0..grid_state.rows() {
        for col in 0..grid_state.cols() {
            if grid_state
                .locked_tiles
                .get([row, col])
                .copied()
                .unwrap_or(0)
                > 1
            {
                found_tiles = true;
                min_row = min_row.min(row);
                max_row = max_row.max(row);
                min_col = min_col.min(col);
                max_col = max_col.max(col);
            }
        }
    }

    found_tiles.then_some(BoundingBox {
        min_row,
        max_row,
        min_col,
        max_col,
    })
}

/// Save the placed tiles, cropped to their extent, as a PNG with empty cells transparent
/// # Errors
/// Fails when no tile is placed, a tile has no colour, or the file cannot be written.
pub fn export_grid_as_png(
    grid_state: &GridState,
    color_mapping: &[[u8; 4]],
    output_path: &str,
) -> crate::io::error::Result<()> {
    use crate::io::error::AlgorithmError;
    let bbox = calculate_bounding_box(grid_state).ok_or(AlgorithmError::InvalidSourceData {
        reason: "No tiles have been placed in the grid".to_string(),
    })?;

    let width = usize_to_u32(bbox.max_col - bbox.min_col + 1);
    let height = usize_to_u32(bbox.max_row - bbox.min_row + 1);

    let mut img = ImageBuffer::new(width, height);

    for row in bbox.min_row..=bbox.max_row {
        for col in bbox.min_col..=bbox.max_col {
            let tile_value = grid_state
                .locked_tiles
                .get([row, col])
                .copied()
                .unwrap_or(0);
            let pixel_x = usize_to_u32(col - bbox.min_col);
            let pixel_y = usize_to_u32(row - bbox.min_row);

            let color = if tile_value > 1 {
                // Tiles: 0=uninitialized, 1=empty, 2+=actual tile
                let color_index = (tile_value - 2) as usize;
                if color_index >= color_mapping.len() {
                    return Err(AlgorithmError::InvalidTileIndex {
                        index: tile_value as usize,
                        max_tiles: color_mapping.len() + 1,
                    });
                }
                let rgba = color_mapping
                    .get(color_index)
                    .copied()
                    .unwrap_or([0, 0, 0, 0]);
                Rgba([rgba[0], rgba[1], rgba[2], rgba[3]])
            } else {
                Rgba([0, 0, 0, 0])
            };

            img.put_pixel(pixel_x, pixel_y, color);
        }
    }

    if let Some(parent) = std::path::Path::new(output_path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| AlgorithmError::FileSystem {
            path: parent.to_path_buf(),
            operation: "create directory",
            source: e,
        })?;
    }

    img.save(output_path)
        .map_err(|e| AlgorithmError::ImageExport {
            path: output_path.into(),
            source: e,
        })?;

    Ok(())
}

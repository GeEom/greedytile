//! Tests for analysis capture with configurable recording radius

use greedytile::io::analysis::AnalysisCapture;
use greedytile::math::convert::usize_to_f64;
use greedytile::spatial::GridState;

// Verifies AnalysisCapture construction and recording functionality with different capture radii
// Verified by breaking capture radius calculations to verify radius affects captured data
#[test]
fn test_analysis_capture_new() {
    let color_mapping = vec![[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
    let grid_extension_radius = 5;

    let analysis = AnalysisCapture::new(color_mapping.clone(), grid_extension_radius);

    let mut grid_state = GridState::new(3, 3, 3);

    if let Some(entropy) = grid_state.entropy.get_mut([1, 1]) {
        *entropy = 0.5;
    }
    if let Some(feasibility) = grid_state.feasibility.get_mut([1, 1]) {
        *feasibility = 0.8;
    }

    let mut analysis = analysis;
    analysis.record_region(1, 1, &grid_state, [0, 0], 0);

    let analysis_radius_0 = AnalysisCapture::new(color_mapping.clone(), 0);
    let analysis_radius_2 = AnalysisCapture::new(color_mapping, 2);

    let mut large_grid = GridState::new(5, 5, 3);

    for i in 0..5 {
        for j in 0..5 {
            if let Some(entropy) = large_grid.entropy.get_mut([i, j]) {
                *entropy = usize_to_f64(i + j) * 0.1;
            }
            if let Some(feasibility) = large_grid.feasibility.get_mut([i, j]) {
                *feasibility = usize_to_f64(i * j) * 0.1;
            }
        }
    }

    let mut analysis_radius_0 = analysis_radius_0;
    let mut analysis_radius_2 = analysis_radius_2;

    analysis_radius_0.record_region(2, 2, &large_grid, [0, 0], 0);

    analysis_radius_2.record_region(2, 2, &large_grid, [0, 0], 0);

    let events_radius_0 = analysis_radius_0.event_count();
    let events_radius_2 = analysis_radius_2.event_count();

    assert_eq!(
        events_radius_0, 1,
        "Radius 0 should capture only center position"
    );
    assert_eq!(
        events_radius_2, 25,
        "Radius 2 should capture 5x5 grid around center"
    );

    assert!(
        events_radius_2 > events_radius_0,
        "Larger radius should capture more events"
    );
}

// Tests that different capture radii can be initialized and used for recording
// Verified by testing boundary conditions and edge cases in grid recording
#[test]
fn test_analysis_capture_radius_initialization() {
    let color_mapping = vec![[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];

    let analysis_small = AnalysisCapture::new(color_mapping.clone(), 0);
    let analysis_medium = AnalysisCapture::new(color_mapping.clone(), 1);
    let analysis_large = AnalysisCapture::new(color_mapping, 2);

    let mut grid_state = GridState::new(10, 10, 3);

    for row in 0..10 {
        for col in 0..10 {
            if let Some(entropy) = grid_state.entropy.get_mut([row, col]) {
                *entropy = usize_to_f64(row).mul_add(0.1, 0.5);
            }
            if let Some(feasibility) = grid_state.feasibility.get_mut([row, col]) {
                *feasibility = usize_to_f64(col).mul_add(-0.05, 0.8);
            }
        }
    }

    let mut analysis_small = analysis_small;
    let mut analysis_medium = analysis_medium;
    let mut analysis_large = analysis_large;

    analysis_small.record_region(5, 5, &grid_state, [0, 0], 0);
    analysis_medium.record_region(5, 5, &grid_state, [0, 0], 0);
    analysis_large.record_region(5, 5, &grid_state, [0, 0], 0);

    analysis_small.record_region(0, 0, &grid_state, [0, 0], 1);
    analysis_medium.record_region(9, 9, &grid_state, [0, 0], 2);
    analysis_large.record_region(1, 8, &grid_state, [0, 0], 3);
}

// Tests that AnalysisCapture handles empty color mappings gracefully
// Verified by modifying array access to test bounds checking with empty color mappings
#[test]
fn test_analysis_capture_empty_color_mapping() {
    let color_mapping = vec![];
    let grid_extension_radius = 5;

    let mut analysis = AnalysisCapture::new(color_mapping, grid_extension_radius);

    let grid_state = GridState::new(2, 2, 1);
    analysis.record_region(0, 0, &grid_state, [0, 0], 0);
}

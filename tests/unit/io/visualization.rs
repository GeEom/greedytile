//! Tests for GIF visualization capture and frame generation

#[cfg(test)]
mod tests {
    use greedytile::io::visualization::VisualizationCapture;

    // Tests VisualizationCapture construction
    // Verified by initializing with non-empty placements
    #[test]
    fn test_visualization_capture_new() {
        let color_mapping = vec![[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
        let viz = VisualizationCapture::new(10, 10, color_mapping, 10000);

        assert_eq!(viz.placement_count(), 0);
    }

    // Tests placement recording increments count
    // Verified by removing record_placement body
    #[test]
    fn test_record_placement() {
        let color_mapping = vec![[255, 0, 0, 255], [0, 255, 0, 255]];
        let mut viz = VisualizationCapture::new(10, 10, color_mapping, 100);

        viz.record_placement(5, 5, 2, 1);
        assert_eq!(viz.placement_count(), 1);

        viz.record_placement(6, 6, 3, 2);
        assert_eq!(viz.placement_count(), 2);
    }

    // Tests error when exporting empty visualization
    // Verified by removing empty placements check
    #[test]
    fn test_export_gif_no_placements() {
        let color_mapping = vec![[255, 0, 0, 255]];
        let viz = VisualizationCapture::new(10, 10, color_mapping, 100);

        let result = viz.export_gif("/dev/null/test.gif", 50);
        assert!(result.is_err());
    }

    // Tests bounds calculation with out-of-bounds placements
    // Verified by ignoring negative coordinates
    #[test]
    fn test_final_bounds_calculation() {
        let color_mapping = vec![[255, 0, 0, 255], [0, 255, 0, 255]];
        let mut viz = VisualizationCapture::new(5, 5, color_mapping, 100);

        viz.record_placement(2, 3, 1, 1);
        viz.record_placement(7, 8, 2, 2);
        viz.record_placement(-2, -3, 1, 3);
        viz.record_placement(10, 10, 2, 4);

        assert_eq!(viz.placement_count(), 4);

        // Use an invalid path that works cross-platform
        let result = viz.export_gif("\0invalid\0path.gif", 50);
        assert!(result.is_err());
    }

    // Tests handling of negative coordinates
    // Verified by ignoring negative removals
    #[test]
    fn test_negative_coordinates() {
        let color_mapping = vec![[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
        let mut viz = VisualizationCapture::new(10, 10, color_mapping, 100);

        viz.record_placement(-5, -3, 1, 1);
        viz.record_placement(-10, 5, 2, 2);
        viz.record_placement(3, -7, 3, 3);
        viz.record_placement(0, 0, 1, 4);
        viz.record_placement(5, 7, 2, 5);
        viz.record_removal(-2, -2, 6);

        assert_eq!(viz.placement_count(), 6);

        // Use an invalid path that works cross-platform
        let result = viz.export_gif("\0invalid\0path.gif", 5);
        assert!(result.is_err());
    }

    // Tests removal recording
    // Verified by recording removals with Some instead of None
    #[test]
    fn test_record_removal() -> crate::TestResult {
        let color_mapping = vec![[255, 0, 0, 255], [0, 255, 0, 255]];
        let mut viz = VisualizationCapture::new(10, 10, color_mapping, 100);

        viz.record_placement(5, 5, 2, 1);
        viz.record_placement(6, 6, 3, 2);
        viz.record_removal(5, 5, 3);
        viz.record_placement(7, 7, 2, 4);
        viz.record_removal(8, 8, 5);

        assert_eq!(viz.placement_count(), 5);

        let placements = viz.get_placements();
        assert_eq!(
            placements.first().ok_or("placement not recorded")?.tile_ref,
            Some(2)
        );
        assert_eq!(
            placements.get(1).ok_or("placement not recorded")?.tile_ref,
            Some(3)
        );
        assert_eq!(
            placements.get(2).ok_or("placement not recorded")?.tile_ref,
            None
        );
        assert_eq!(
            placements.get(3).ok_or("placement not recorded")?.tile_ref,
            Some(2)
        );
        assert_eq!(
            placements.get(4).ok_or("placement not recorded")?.tile_ref,
            None
        );

        let removal = placements.get(2).ok_or("placement not recorded")?;
        assert_eq!(removal.row, 5);
        assert_eq!(removal.col, 5);
        assert_eq!(removal.iteration, 3);
        Ok(())
    }
}

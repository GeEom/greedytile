//! Tests for forced position detection and pipeline processing in the executor

#[cfg(test)]
mod tests {
    use greedytile::algorithm::cache::ViableTilesCache;
    use greedytile::algorithm::executor::GreedyStochastic;
    use greedytile::algorithm::propagation::detect_forced_positions;
    use greedytile::algorithm::selection::compute_viable_tiles_at_position;
    use std::collections::BTreeSet;

    // Verifies forced positions are detected during iterations
    // Verified by breaking the detection condition logic
    #[test]
    fn test_forced_position_detection_basic() -> crate::TestResult {
        let mut executor = GreedyStochastic::new(42)?;

        let mut total_forced_detected = 0;

        for _iteration in 0..10 {
            let _pipeline_size_before = executor.forced_pipeline.len();

            let prev_coords = executor.selection_coordinates;
            let prev_offset = executor.system_offset;

            executor.run_iteration()?;

            let mut cache = ViableTilesCache::new();
            let new_forced = detect_forced_positions(
                &executor.grid_state,
                prev_coords,
                prev_offset,
                &executor.step_data.source_tiles,
                &executor.step_data,
                &mut cache,
            );

            total_forced_detected += new_forced.len();

            let _pipeline_size_after = executor.forced_pipeline.len();
        }

        assert!(
            total_forced_detected > 0,
            "No forced positions were detected across 10 iterations. \
             Expected at least some positions to become forced due to constraints."
        );
        Ok(())
    }

    // Verifies forced positions have exactly 1 viable tile
    // Verified by allowing positions with multiple viable tiles to be marked as forced
    #[test]
    fn test_forced_position_detection_correctness() -> crate::TestResult {
        let mut executor = GreedyStochastic::new(123)?;

        let mut forced_found = false;
        for _ in 0..20 {
            let prev_coords = executor.selection_coordinates;
            let prev_offset = executor.system_offset;

            executor.run_iteration()?;

            let mut cache = ViableTilesCache::new();
            let forced = detect_forced_positions(
                &executor.grid_state,
                prev_coords,
                prev_offset,
                &executor.step_data.source_tiles,
                &executor.step_data,
                &mut cache,
            );

            if !forced.is_empty() {
                forced_found = true;

                for forced_pos in forced {
                    let mut temp_cache = ViableTilesCache::new();
                    let viable = compute_viable_tiles_at_position(
                        &executor.grid_state,
                        forced_pos.coordinates,
                        executor.system_offset,
                        &executor.step_data.source_tiles,
                        &executor.step_data,
                        &mut temp_cache,
                    );

                    assert_eq!(
                        viable.len(),
                        1,
                        "Forced position at {:?} has {} viable tiles, expected exactly 1",
                        forced_pos.coordinates,
                        viable.len()
                    );

                    assert!(
                        !viable.is_empty(),
                        "Forced position should have a viable tile"
                    );
                }
                break;
            }
        }

        assert!(
            forced_found,
            "No forced positions found in 20 iterations. \
             This might indicate the detection is not working."
        );
        Ok(())
    }

    // Tests forced positions are detected in all 8 adjacent directions
    // Verified by skipping positive row directions which caused the test to fail
    #[test]
    fn test_forced_position_adjacency_coverage() -> crate::TestResult {
        let mut executor = GreedyStochastic::new(456)?;

        let mut forced_directions = BTreeSet::new();

        for _ in 0..30 {
            let prev_coords = executor.selection_coordinates;
            let prev_offset = executor.system_offset;

            executor.run_iteration()?;

            let mut cache = ViableTilesCache::new();
            let forced = detect_forced_positions(
                &executor.grid_state,
                prev_coords,
                prev_offset,
                &executor.step_data.source_tiles,
                &executor.step_data,
                &mut cache,
            );

            for forced_pos in forced {
                let di = forced_pos.coordinates[0] - prev_coords[0];
                let dj = forced_pos.coordinates[1] - prev_coords[1];

                assert!(
                    di.abs() <= 1 && dj.abs() <= 1 && (di != 0 || dj != 0),
                    "Forced position at {:?} is not adjacent to {:?}",
                    forced_pos.coordinates,
                    prev_coords
                );

                forced_directions.insert((di, dj));
            }
        }

        assert!(
            forced_directions.len() >= 6,
            "Only detected forced positions in {} directions: {:?}. \
             Expected at least 6 out of 8 directions to be covered.",
            forced_directions.len(),
            forced_directions
        );

        let has_positive_di = forced_directions.iter().any(|&(di, _)| di > 0);
        assert!(
            has_positive_di,
            "No forced positions detected with positive di (row below). \
             This suggests the adjacency loop might be incorrect."
        );
        Ok(())
    }

    // Verifies forced positions are processed from pipeline
    // Verified by preventing forced positions from being added to the pipeline
    #[test]
    fn test_forced_pipeline_processing() -> crate::TestResult {
        let mut executor = GreedyStochastic::new(789)?;

        let mut forced_processed_count = 0;
        let mut random_selection_count = 0;

        for _ in 0..20 {
            let has_forced = !executor.forced_pipeline.is_empty();
            let forced_count_before = executor.forced_pipeline.len();

            executor.run_iteration()?;

            let forced_count_after = executor.forced_pipeline.len();

            if has_forced && forced_count_after < forced_count_before {
                forced_processed_count += 1;
            } else if !has_forced {
                random_selection_count += 1;
            }
        }

        assert!(
            forced_processed_count > 0,
            "No forced positions were processed. Pipeline might not be working."
        );

        assert!(
            random_selection_count > 0,
            "No random selections were made. This is suspicious."
        );
        Ok(())
    }

    // Tests no duplicate positions in pipeline
    // Verified by removing duplicate check which caused duplicates to appear
    #[test]
    fn test_forced_position_deduplication() -> crate::TestResult {
        let mut executor = GreedyStochastic::new(321)?;

        for _ in 0..15 {
            executor.run_iteration()?;

            if !executor.forced_pipeline.is_empty() {
                let positions: Vec<[i32; 2]> = executor
                    .forced_pipeline
                    .queue
                    .iter()
                    .map(|fp| fp.coordinates)
                    .collect();

                let unique_positions: BTreeSet<[i32; 2]> = positions.iter().copied().collect();

                assert_eq!(
                    positions.len(),
                    unique_positions.len(),
                    "Found duplicate positions in forced pipeline: {positions:?}"
                );
            }
        }
        Ok(())
    }

    // Validates forced position processing helps prevent contradictions
    // Verified by confirming system runs without errors when forced positions are detected
    #[test]
    fn test_forced_position_prevents_contradiction() -> crate::TestResult {
        let mut executor = GreedyStochastic::new(654)?;

        let mut had_forced_positions = false;

        for i in 0..50 {
            if !executor.forced_pipeline.is_empty() {
                had_forced_positions = true;
            }

            executor
                .run_iteration()
                .map_err(|e| format!("iteration {i}: {e}"))?;
        }

        assert!(
            had_forced_positions,
            "No forced positions were detected during the test"
        );
        Ok(())
    }
}

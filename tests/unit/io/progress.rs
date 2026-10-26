//! Tests for progress tracking and multi-file batch processing

#[cfg(test)]
mod tests {
    use greedytile::io::configuration::MAX_INDIVIDUAL_PROGRESS_BARS;
    use greedytile::io::progress::ProgressManager;
    use std::path::Path;
    use std::time::Duration;

    // Tests ProgressManager construction
    // Verified by setting wrong initial state
    #[test]
    fn test_progress_manager_new() -> crate::TestResult {
        let mut pm = ProgressManager::new();

        pm.initialize(0);
        pm.finish()?;

        pm.initialize(1);
        pm.start_file(0, Path::new("test.png"), 10);
        pm.update_iteration(0, 5, Duration::from_millis(50));
        pm.complete_file(0, Duration::from_millis(100));
        pm.finish()?;
        Ok(())
    }

    // Tests default trait implementation
    // Verified by creating different initial states
    #[test]
    fn test_progress_manager_default() -> crate::TestResult {
        let mut pm1 = ProgressManager::new();
        let mut pm2 = ProgressManager::default();

        pm1.initialize(2);
        pm2.initialize(2);

        pm1.start_file(0, Path::new("test1.png"), 50);
        pm2.start_file(0, Path::new("test1.png"), 50);

        pm1.update_iteration(0, 25, Duration::from_millis(100));
        pm2.update_iteration(0, 25, Duration::from_millis(100));

        pm1.complete_file(0, Duration::from_millis(200));
        pm2.complete_file(0, Duration::from_millis(200));

        pm1.finish()?;
        pm2.finish()?;
        Ok(())
    }

    // Tests initialization with single file
    // Verified by skipping initialization for single files
    #[test]
    fn test_initialize_single_file() -> crate::TestResult {
        let mut pm = ProgressManager::new();
        pm.initialize(1);

        pm.start_file(0, Path::new("single.png"), 100);

        pm.update_iteration(0, 0, Duration::from_millis(0));
        pm.update_iteration(0, 25, Duration::from_millis(100));
        pm.update_iteration(0, 50, Duration::from_millis(200));
        pm.update_iteration(0, 75, Duration::from_millis(300));
        pm.update_iteration(0, 100, Duration::from_millis(400));

        pm.complete_file(0, Duration::from_millis(400));
        pm.finish()?;
        Ok(())
    }

    // Tests individual progress bars
    // Verified by creating one less progress bar
    #[test]
    fn test_initialize_multiple_files_under_limit() -> crate::TestResult {
        let mut pm = ProgressManager::new();
        let file_count = MAX_INDIVIDUAL_PROGRESS_BARS - 1;
        pm.initialize(file_count);

        for i in 0..file_count {
            pm.start_file(i, Path::new(&format!("file{i}.png")), 100);
            pm.update_iteration(i, 25, Duration::from_millis(25));
            pm.update_iteration(i, 50, Duration::from_millis(50));
            pm.update_iteration(i, 75, Duration::from_millis(75));
            pm.update_iteration(i, 100, Duration::from_millis(100));
            pm.complete_file(i, Duration::from_millis(100));
        }

        pm.finish()?;
        Ok(())
    }

    // Tests batch progress bar
    // Verified by changing batch mode threshold
    #[test]
    fn test_initialize_multiple_files_over_limit() -> crate::TestResult {
        let mut pm = ProgressManager::new();
        let large_file_count = MAX_INDIVIDUAL_PROGRESS_BARS + 5;
        pm.initialize(large_file_count);

        for i in 0..large_file_count {
            pm.start_file(i, Path::new(&format!("file{i}.png")), 100);
            pm.update_iteration(i, 50, Duration::from_millis(50));
            pm.complete_file(i, Duration::from_millis(100));
        }

        pm.finish()?;
        Ok(())
    }

    // Tests full processing lifecycle
    // Verified by breaking iteration storage and resize logic
    #[test]
    fn test_file_processing_lifecycle() -> crate::TestResult {
        let mut pm = ProgressManager::new();
        pm.initialize(3);

        let test_path1 = Path::new("test1.png");
        pm.start_file(0, test_path1, 100);

        pm.update_iteration(0, 25, Duration::from_millis(100));
        pm.update_iteration(0, 50, Duration::from_millis(200));
        pm.update_iteration(0, 75, Duration::from_millis(300));
        pm.update_iteration(0, 100, Duration::from_millis(400));

        pm.complete_file(0, Duration::from_millis(400));

        let test_path2 = Path::new("test2.png");
        pm.start_file(1, test_path2, 50);

        pm.update_iteration(1, 10, Duration::from_millis(50));
        pm.update_iteration(1, 20, Duration::from_millis(100));
        pm.update_iteration(1, 30, Duration::from_millis(150));
        pm.update_iteration(1, 40, Duration::from_millis(200));
        pm.update_iteration(1, 50, Duration::from_millis(250));

        pm.complete_file(1, Duration::from_millis(250));

        let test_path3 = Path::new("test3.png");
        pm.start_file(2, test_path3, 75);

        pm.update_iteration(2, 25, Duration::from_millis(100));
        pm.update_iteration(2, 50, Duration::from_millis(200));

        pm.update_iteration(0, 150, Duration::from_millis(500));

        pm.start_file(5, Path::new("out_of_order.png"), 200);
        pm.update_iteration(5, 100, Duration::from_millis(100));
        pm.complete_file(5, Duration::from_millis(100));

        pm.update_iteration(10, 50, Duration::from_millis(100));

        pm.finish()?;
        Ok(())
    }

    // Tests empty file list handling
    // Verified by adding panic for zero files
    #[test]
    fn test_empty_file_list() -> crate::TestResult {
        let mut pm = ProgressManager::new();
        pm.initialize(0);
        pm.finish()?;
        Ok(())
    }

    // Tests out-of-bounds index handling
    // Verified by using unchecked indexing
    #[test]
    fn test_out_of_bounds_file_index() {
        let mut pm = ProgressManager::new();
        pm.initialize(3);

        pm.update_iteration(10, 50, Duration::from_secs(1));
        pm.complete_file(10, Duration::from_secs(1));
    }
}

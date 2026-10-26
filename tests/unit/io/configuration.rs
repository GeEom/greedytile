//! Tests for algorithm configuration constants and validation

#[cfg(test)]
mod tests {
    use greedytile::io::configuration::{
        ADJACENCY_CANDIDATES_CONSIDERED, CANDIDATES_CONSIDERED, DEFAULT_MAX_ITERATIONS,
        DEFAULT_SEED, GIF_FRAME_DELAY_MS, MAX_GRID_DIMENSION, MAX_INDIVIDUAL_PROGRESS_BARS,
        OUTPUT_SUFFIX, PROGRESS_BAR_WIDTH, TILE_SIZE,
    };

    // Tests candidate values are correct
    // Verified by changing constant values
    #[test]
    fn test_candidates_considered_values() {
        assert_eq!(CANDIDATES_CONSIDERED, 15);
        assert_eq!(ADJACENCY_CANDIDATES_CONSIDERED, 30);
    }

    // Tests tile size is odd number
    // Verified by changing to even number
    #[test]
    fn test_tile_size_value() {
        assert_eq!(TILE_SIZE, 3);
    }

    // Tests maximum grid dimension value
    // Verified by reducing dimension limit
    #[test]
    fn test_max_grid_dimension() {
        assert_eq!(MAX_GRID_DIMENSION, 10_000);
    }

    // Tests adjacency candidates exceed standard candidates
    // Verified by inverting relationship values
    #[test]
    fn test_constants_relationship() {
        assert_eq!(ADJACENCY_CANDIDATES_CONSIDERED, 30);
        assert_eq!(CANDIDATES_CONSIDERED, 15);
    }

    // Tests progress bar limit
    // Verified by increasing bar limit
    #[test]
    fn test_max_progress_bars_value() {
        assert_eq!(MAX_INDIVIDUAL_PROGRESS_BARS, 5);
    }

    // Tests progress bar width
    // Verified by changing width value
    #[test]
    fn test_progress_bar_width() {
        assert_eq!(PROGRESS_BAR_WIDTH, 50);
    }

    // Tests default seed is fixed
    // Verified by changing seed value
    #[test]
    fn test_default_seed_is_reproducible() {
        assert_eq!(DEFAULT_SEED, 42);
    }

    // Tests default iteration count
    // Verified by reducing iteration count
    #[test]
    fn test_default_iterations_is_reasonable() {
        assert_eq!(DEFAULT_MAX_ITERATIONS, 1000);
    }

    // Tests output suffix starts with underscore
    // Verified by removing underscore prefix
    #[test]
    fn test_output_suffix_format() {
        assert!(OUTPUT_SUFFIX.starts_with('_'));
        assert!(OUTPUT_SUFFIX.len() < 20);
    }

    // Tests filesystem safety of suffix
    // Verified by adding special character
    #[test]
    fn test_output_suffix_no_special_chars() {
        for ch in OUTPUT_SUFFIX.chars() {
            assert!(
                ch.is_alphanumeric() || ch == '_' || ch == '-',
                "Output suffix contains invalid character: {ch}"
            );
        }
    }

    // Tests GIF frame delay value
    // Verified by changing delay value
    #[test]
    fn test_gif_frame_delay() {
        assert_eq!(GIF_FRAME_DELAY_MS, 5);
    }
}

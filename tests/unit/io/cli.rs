//! Tests for command-line interface parsing and file processing

#[cfg(test)]
mod tests {
    use clap::Parser;
    use greedytile::io::cli::Cli;
    use greedytile::io::configuration::{DEFAULT_MAX_ITERATIONS, DEFAULT_SEED};
    use std::path::PathBuf;

    // Tests CLI parsing with only required target file argument
    // Verified by changing default values to ensure defaults are used
    #[test]
    fn test_cli_parse_minimal_args() {
        let args = vec!["program", "test.png"];
        let cli = Cli::parse_from(args);

        assert_eq!(cli.target, PathBuf::from("test.png"));
        assert_eq!(cli.seed, DEFAULT_SEED);
        assert_eq!(cli.iterations, DEFAULT_MAX_ITERATIONS);
        assert!(!cli.quiet);
    }

    // Tests CLI parsing with all available arguments
    // Verified by modifying custom parsers to ensure they're invoked
    #[test]
    fn test_cli_parse_all_args() {
        let args = vec![
            "program",
            "input.png",
            "--seed",
            "123",
            "--iterations",
            "500",
            "--quiet",
            "--no-skip",
        ];
        let cli = Cli::parse_from(args);

        assert_eq!(cli.target, PathBuf::from("input.png"));
        assert_eq!(cli.seed, 123);
        assert_eq!(cli.iterations, 500);
        assert!(cli.quiet);
    }

    // Tests file skip behavior based on --no-skip flag
    // Verified by inverting boolean logic in skip_existing method
    #[test]
    fn test_skip_existing_logic() {
        let args_default = vec!["program", "test.png"];
        let cli_default = Cli::parse_from(args_default);
        assert!(cli_default.skip_existing());

        let args_no_skip = vec!["program", "test.png", "--no-skip"];
        let cli_no_skip = Cli::parse_from(args_no_skip);
        assert!(!cli_no_skip.skip_existing());
    }

    // Tests progress display based on --quiet flag
    // Verified by inverting quiet flag logic
    #[test]
    fn test_should_show_progress() {
        let args_default = vec!["program", "test.png"];
        let cli_default = Cli::parse_from(args_default);
        assert!(cli_default.should_show_progress());

        let args_quiet = vec!["program", "test.png", "--quiet"];
        let cli_quiet = Cli::parse_from(args_quiet);
        assert!(!cli_quiet.should_show_progress());
    }

    // Tests short flag parsing (-s, -i)
    // Verified by changing short flag definitions
    #[test]
    fn test_cli_short_flags() {
        let args = vec!["program", "test.png", "-s", "999", "-i", "100"];
        let cli = Cli::parse_from(args);

        assert_eq!(cli.seed, 999);
        assert_eq!(cli.iterations, 100);
    }

    use greedytile::io::cli::FileProcessor;
    use std::fs;
    use tempfile::TempDir;

    // Tests FileProcessor construction
    // Verified by modifying constructor logic
    #[test]
    fn test_file_processor_new() {
        let cli = create_test_cli("test.png");
        let _processor = FileProcessor::new(cli);
    }

    // Tests error handling for missing files
    // Verified by removing error return for nonexistent files
    #[test]
    fn test_process_nonexistent_file() {
        let cli = create_test_cli("nonexistent.png");
        let mut processor = FileProcessor::new(cli);

        let result = processor.process();
        assert!(result.is_err());
    }

    // Tests error handling for non-PNG files
    // Verified by removing file type validation
    #[test]
    fn test_process_invalid_file_type() -> crate::TestResult {
        let temp_dir = TempDir::new()?;
        let txt_file = temp_dir.path().join("test.txt");
        fs::write(&txt_file, "not a png")?;

        let cli = create_test_cli(txt_file.to_str().ok_or("path is not UTF-8")?);
        let mut processor = FileProcessor::new(cli);

        let result = processor.process();
        assert!(result.is_err());
        Ok(())
    }

    // Tests skip logic when output file exists
    // Verified by removing skip check
    #[test]
    fn test_skip_existing_file() -> crate::TestResult {
        let temp_dir = TempDir::new()?;
        let input_file = temp_dir.path().join("test.png");
        let output_file = temp_dir.path().join("test_result.png");

        fs::write(&input_file, "fake png")?;
        fs::write(&output_file, "fake png")?;

        let cli = create_test_cli(input_file.to_str().ok_or("path is not UTF-8")?);
        let mut processor = FileProcessor::new(cli);

        let result = processor.process();
        assert!(result.is_ok());
        Ok(())
    }

    // Tests processing empty directories
    // Verified by adding error for empty directories
    #[test]
    fn test_process_empty_directory() -> crate::TestResult {
        let temp_dir = TempDir::new()?;

        let cli = create_test_cli(temp_dir.path().to_str().ok_or("path is not UTF-8")?);
        let mut processor = FileProcessor::new(cli);

        let result = processor.process();
        assert!(result.is_ok());
        Ok(())
    }

    // Tests output filename generation with suffix
    // Verified by changing output suffix to verify path generation
    #[test]
    fn test_output_path_generation() -> crate::TestResult {
        let temp_dir = tempfile::tempdir()?;
        let input_file = temp_dir.path().join("test_image.png");

        fs::write(&input_file, "fake png")?;

        let output_file = temp_dir.path().join("test_image_result.png");
        fs::write(&output_file, "output")?;

        let cli = create_test_cli(input_file.to_str().ok_or("path is not UTF-8")?);
        let mut processor = FileProcessor::new(cli);

        let result = processor.process();
        assert!(result.is_ok());

        let input_file2 = temp_dir.path().join("test_image2.png");
        fs::write(&input_file2, "fake png")?;

        let cli2 = create_test_cli(input_file2.to_str().ok_or("path is not UTF-8")?);
        let mut processor2 = FileProcessor::new(cli2);

        assert!(processor2.process().is_err(), "fake PNG data cannot decode");

        let wrong_output = temp_dir.path().join("test_image2_output.png");
        assert!(
            !wrong_output.exists(),
            "Should not create file with wrong suffix"
        );
        Ok(())
    }

    // Tests quiet mode configuration and behavior
    // Verified by testing quiet flag affects progress display
    #[test]
    fn test_quiet_mode() -> crate::TestResult {
        let temp_dir = tempfile::tempdir()?;
        let input_file = temp_dir.path().join("test.png");

        fs::write(&input_file, "fake png")?;

        let args_quiet = vec![
            "program",
            input_file.to_str().ok_or("path is not UTF-8")?,
            "--quiet",
        ];
        let cli_quiet = Cli::parse_from(args_quiet);
        assert!(cli_quiet.quiet, "Quiet flag should be set");
        assert!(
            !cli_quiet.should_show_progress(),
            "Should not show progress in quiet mode"
        );

        let mut processor_quiet = FileProcessor::new(cli_quiet);
        assert!(processor_quiet.process().is_err());

        let args_normal = vec!["program", input_file.to_str().ok_or("path is not UTF-8")?];
        let cli_normal = Cli::parse_from(args_normal);
        assert!(!cli_normal.quiet, "Quiet flag should not be set by default");
        assert!(
            cli_normal.should_show_progress(),
            "Should show progress by default"
        );

        let mut processor_normal = FileProcessor::new(cli_normal);
        assert!(processor_normal.process().is_err());
        Ok(())
    }

    fn create_test_cli(target: &str) -> Cli {
        let args = vec!["program", target];
        Cli::parse_from(args)
    }

    // Tests width/height CLI arguments create proper bounds configuration
    // Verified by changing short flag from 'w' to 'x'
    #[test]
    fn test_cli_dimension_arguments() {
        let args_width = vec!["program", "test.png", "-w", "512"];
        let cli_width = Cli::parse_from(args_width);
        assert_eq!(cli_width.width, Some(512));
        assert_eq!(cli_width.height, None);

        let args_height = vec!["program", "test.png", "-H", "256"];
        let cli_height = Cli::parse_from(args_height);
        assert_eq!(cli_height.width, None);
        assert_eq!(cli_height.height, Some(256));

        let args_both = vec!["program", "test.png", "-w", "512", "-H", "256"];
        let cli_both = Cli::parse_from(args_both);
        assert_eq!(cli_both.width, Some(512));
        assert_eq!(cli_both.height, Some(256));
    }

    // Tests multiple feature flags can be combined correctly
    // Verified by removing short flag from visualize option
    #[test]
    fn test_cli_feature_flag_combinations() {
        let args = vec![
            "program",
            "test.png",
            "--visualize",
            "--analysis",
            "--prefill",
            "--rotate",
            "--mirror",
        ];
        let cli = Cli::parse_from(args);

        assert!(cli.outputs.visualize);
        assert!(cli.outputs.analysis);
        assert!(cli.tiles.prefill);
        assert!(cli.tiles.rotate);
        assert!(cli.tiles.mirror);

        let args_short = vec!["program", "test.png", "-v", "-a", "-p", "-r", "-m"];
        let cli_short = Cli::parse_from(args_short);

        assert!(cli_short.outputs.visualize);
        assert!(cli_short.outputs.analysis);
        assert!(cli_short.tiles.prefill);
        assert!(cli_short.tiles.rotate);
        assert!(cli_short.tiles.mirror);
    }
}

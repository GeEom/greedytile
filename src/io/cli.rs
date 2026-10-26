//! Command-line interface for batch processing PNG files with pattern generation

use crate::algorithm::executor::{AlgorithmConfig, GreedyStochastic};
use crate::analysis::patterns::ImageProcessor;
use crate::io::configuration::{
    ADJACENCY_CANDIDATES_CONSIDERED, CANDIDATES_CONSIDERED, DEFAULT_MAX_ITERATIONS, DEFAULT_SEED,
    GRID_EXTENSION_RADIUS, OUTPUT_SUFFIX, PATTERN_INFLUENCE_DISTANCE, TILE_SIZE,
};
use crate::io::error::Result;
use crate::io::image::export_grid_as_png;
use crate::io::prefill::PrefillData;
use crate::io::progress::ProgressManager;
use clap::{Args, Parser};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "infotiles")]
#[command(
    author,
    version,
    about = "Generate tile patterns using random greedy algorithm"
)]
/// Command-line arguments for the pattern generation tool
pub struct Cli {
    /// Input PNG file or directory to process
    #[arg(value_name = "TARGET")]
    pub target: PathBuf,

    /// Random seed for reproducible generation
    #[arg(short, long, default_value_t = DEFAULT_SEED)]
    pub seed: u64,

    /// Maximum iterations before stopping
    #[arg(short, long, default_value_t = DEFAULT_MAX_ITERATIONS)]
    pub iterations: usize,

    /// Suppress progress output
    #[arg(short, long)]
    pub quiet: bool,

    /// Process files even if output exists
    #[arg(short, long)]
    pub no_skip: bool,

    /// Maximum width in pixels (implies square if height not specified)
    #[arg(short = 'w', long)]
    pub width: Option<usize>,

    /// Maximum height in pixels
    #[arg(short = 'H', long)]
    pub height: Option<usize>,

    /// Tile set and prefill flags
    #[command(flatten)]
    pub tiles: TileOptions,

    /// Extra output flags
    #[command(flatten)]
    pub outputs: OutputOptions,
}

/// Flags that shape the tile set and the starting grid
#[derive(Args, Clone, Copy, Debug, Default)]
pub struct TileOptions {
    /// Use prefill image if available (looks for <input>_pre.png)
    #[arg(short, long)]
    pub prefill: bool,

    /// Enable tile rotation transformations (90°, 180°, 270°)
    #[arg(short = 'r', long)]
    pub rotate: bool,

    /// Enable tile mirroring transformations (horizontal reflection)
    #[arg(short = 'm', long)]
    pub mirror: bool,
}

/// Flags that add files beside the generated image
#[derive(Args, Clone, Copy, Debug, Default)]
pub struct OutputOptions {
    /// Enable visualization output as animated GIF
    #[arg(short, long)]
    pub visualize: bool,

    /// Enable analysis capture and export
    #[arg(short, long)]
    pub analysis: bool,
}

impl Cli {
    /// Check if existing output files should be skipped
    #[must_use]
    pub const fn skip_existing(&self) -> bool {
        !self.no_skip
    }

    /// Check if progress should be displayed
    #[must_use]
    pub const fn should_show_progress(&self) -> bool {
        !self.quiet
    }
}

/// Orchestrates batch processing of PNG files with progress tracking
pub struct FileProcessor {
    cli: Cli,
    progress_manager: Option<ProgressManager>,
}

impl FileProcessor {
    /// Create a new file processor with the given CLI arguments
    pub fn new(cli: Cli) -> Self {
        let progress_manager = cli.should_show_progress().then(ProgressManager::new);

        Self {
            cli,
            progress_manager,
        }
    }

    /// Process files according to CLI arguments
    /// # Errors
    /// Fails when the target is not a PNG or directory, or a file fails to process.
    pub fn process(&mut self) -> Result<()> {
        let files = self.collect_files()?;

        if files.is_empty() {
            return Ok(());
        }

        if let Some(ref mut pm) = self.progress_manager {
            pm.initialize(files.len());
        }

        for (index, file) in files.iter().enumerate() {
            self.process_file(file, index)?;
        }

        if let Some(ref mut pm) = self.progress_manager {
            pm.finish()?;
        }

        Ok(())
    }

    fn collect_files(&self) -> Result<Vec<PathBuf>> {
        if self.cli.target.is_file() {
            if self.cli.target.extension().and_then(|s| s.to_str()) == Some("png") {
                if self.should_process_file(&self.cli.target)? {
                    Ok(vec![self.cli.target.clone()])
                } else {
                    Ok(vec![])
                }
            } else {
                Err(crate::io::error::io_error(
                    "Target file must be a PNG image",
                ))
            }
        } else if self.cli.target.is_dir() {
            let mut files = Vec::new();
            for entry in std::fs::read_dir(&self.cli.target)? {
                let path = entry?.path();
                if path.extension().and_then(|s| s.to_str()) == Some("png")
                    && self.should_process_file(&path)?
                {
                    files.push(path);
                }
            }
            files.sort();
            Ok(files)
        } else {
            Err(crate::io::error::io_error(
                "Target must be a PNG file or directory",
            ))
        }
    }

    fn should_process_file(&self, input_path: &Path) -> Result<bool> {
        if !self.cli.skip_existing() {
            return Ok(true);
        }

        let output_path = Self::get_output_path(input_path);
        if output_path.exists() {
            if !self.cli.quiet {
                writeln!(
                    std::io::stderr().lock(),
                    "Skipping: {} (output exists)",
                    input_path.display()
                )?;
            }
            Ok(false)
        } else {
            Ok(true)
        }
    }

    fn process_file(&mut self, input_path: &Path, index: usize) -> Result<()> {
        let start_time = Instant::now();
        let output_path = Self::get_output_path(input_path);

        if let Some(ref mut pm) = self.progress_manager {
            pm.start_file(index, input_path, self.cli.iterations);
        }

        let image_processor = ImageProcessor::from_png_path(input_path)?;

        let bounds = match (self.cli.height, self.cli.width) {
            (Some(h), Some(w)) => Some((h, w)),
            (Some(h), None) => Some((h, h)),
            (None, Some(w)) => Some((w, w)),
            (None, None) => None,
        };

        let config = AlgorithmConfig {
            candidates_considered: CANDIDATES_CONSIDERED,
            adjacency_candidates_considered: ADJACENCY_CANDIDATES_CONSIDERED,
            pattern_influence_distance: PATTERN_INFLUENCE_DISTANCE,
            grid_extension_radius: GRID_EXTENSION_RADIUS,
            tile_size: TILE_SIZE,
            include_rotations: self.cli.tiles.rotate,
            include_reflections: self.cli.tiles.mirror,
            bounds,
        };

        let mut executor =
            GreedyStochastic::from_image_processor(image_processor, config, self.cli.seed)?;

        if self.cli.tiles.prefill {
            let prefill_path = Self::get_prefill_path(input_path);
            if prefill_path.exists() {
                let prefill_data = PrefillData::from_png(&prefill_path, executor.color_mapping())?;
                executor.apply_prefill(prefill_data)?;
            } else if !self.cli.quiet {
                writeln!(
                    std::io::stderr().lock(),
                    "No prefill found at: {} (continuing without prefill)",
                    prefill_path.display()
                )?;
            }
        }

        // Enable visualization if requested or if analysis is requested
        if self.cli.outputs.visualize || self.cli.outputs.analysis {
            executor.enable_visualization(self.cli.iterations);
        }

        if self.cli.outputs.analysis {
            executor.enable_analysis();
        }

        for iteration in 1..=self.cli.iterations {
            if let Some(ref mut pm) = self.progress_manager {
                pm.update_iteration(index, iteration, start_time.elapsed());
            }

            let should_continue = executor.execute_iteration()?;
            if !should_continue {
                break;
            }
        }

        export_grid_as_png(
            executor.grid_state(),
            executor.color_mapping(),
            output_path
                .to_str()
                .ok_or_else(|| crate::io::error::io_error("Invalid output path"))?,
        )?;

        if self.cli.outputs.visualize {
            let viz_path = Self::get_visualization_path(input_path);
            executor.export_visualization(
                viz_path
                    .to_str()
                    .ok_or_else(|| crate::io::error::io_error("Invalid visualization path"))?,
            )?;
        }

        if self.cli.outputs.analysis {
            let analysis_path = Self::get_analysis_path(input_path);
            if let (Some(viz), Some(analysis)) = (&executor.visualization, &executor.analysis) {
                analysis.export_analysis(
                    viz,
                    analysis_path
                        .to_str()
                        .ok_or_else(|| crate::io::error::io_error("Invalid analysis path"))?,
                    crate::io::configuration::GIF_FRAME_DELAY_MS,
                )?;
            }
        }

        if let Some(ref mut pm) = self.progress_manager {
            pm.complete_file(index, start_time.elapsed());
        }

        Ok(())
    }

    fn get_prefill_path(input_path: &Path) -> PathBuf {
        let stem = input_path.file_stem().unwrap_or_default();
        let prefill_name = format!("{}_pre.png", stem.to_string_lossy());

        if let Some(parent) = input_path.parent() {
            parent.join(prefill_name)
        } else {
            PathBuf::from(prefill_name)
        }
    }

    fn get_output_path(input_path: &Path) -> PathBuf {
        let stem = input_path.file_stem().unwrap_or_default();
        let extension = input_path.extension().unwrap_or_default();
        let output_name = format!(
            "{}{}.{}",
            stem.to_string_lossy(),
            OUTPUT_SUFFIX,
            extension.to_string_lossy()
        );

        if let Some(parent) = input_path.parent() {
            parent.join(output_name)
        } else {
            PathBuf::from(output_name)
        }
    }

    fn get_visualization_path(input_path: &Path) -> PathBuf {
        let stem = input_path.file_stem().unwrap_or_default();
        let viz_name = format!("{}_visualization.gif", stem.to_string_lossy());

        if let Some(parent) = input_path.parent() {
            parent.join(viz_name)
        } else {
            PathBuf::from(viz_name)
        }
    }

    fn get_analysis_path(input_path: &Path) -> PathBuf {
        let stem = input_path.file_stem().unwrap_or_default();
        let analysis_name = format!("{}_analysis.gif", stem.to_string_lossy());

        if let Some(parent) = input_path.parent() {
            parent.join(analysis_name)
        } else {
            PathBuf::from(analysis_name)
        }
    }
}

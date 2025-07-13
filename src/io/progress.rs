//! Multi-file progress tracking with automatic batching for large sets

use crate::io::configuration::MAX_INDIVIDUAL_PROGRESS_BARS;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::path::Path;
use std::sync::LazyLock;
use std::time::Duration;

/// Coordinates progress display for batch operations
///
/// Automatically switches between individual progress bars (for small batches)
/// and a single batch progress bar (for large batches) based on file count
pub struct ProgressManager {
    multi_progress: MultiProgress,
    batch_bar: Option<ProgressBar>,
    file_bars: Vec<ProgressBar>,
    file_count: usize,
    /// Stores (`filename`, `current_iter`, `max_iter`) for rolling window display
    file_states: Vec<(String, usize, usize)>,
}

impl Default for ProgressManager {
    fn default() -> Self {
        Self::new()
    }
}

static PROGRESS_STYLE: LazyLock<ProgressStyle> = LazyLock::new(|| {
    ProgressStyle::default_bar()
        .template("{msg} [{bar:30.cyan/blue}] {prefix}")
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("█▉▊▋▌▍▎▏ ")
});

static BATCH_STYLE: LazyLock<ProgressStyle> = LazyLock::new(|| {
    ProgressStyle::default_bar()
        .template("[{elapsed_precise}] Files: [{bar:40.cyan/blue}] {pos}/{len}")
        .unwrap_or_else(|_| ProgressStyle::default_bar())
});

impl ProgressManager {
    /// Create a new progress manager
    pub fn new() -> Self {
        Self {
            multi_progress: MultiProgress::new(),
            batch_bar: None,
            file_bars: Vec::new(),
            file_count: 0,
            file_states: Vec::new(),
        }
    }

    /// Initialize progress bars based on file count
    ///
    /// # Panics
    ///
    /// Panics if the progress bar template is invalid (this should never happen
    /// with the hardcoded templates used)
    pub fn initialize(&mut self, file_count: usize) {
        self.file_count = file_count;

        // Switch to batch mode for large file sets to avoid terminal spam
        if file_count > MAX_INDIVIDUAL_PROGRESS_BARS + 1 {
            let batch_bar = ProgressBar::new(file_count as u64);
            batch_bar.set_style(BATCH_STYLE.clone());
            self.batch_bar = Some(self.multi_progress.add(batch_bar));
        }
    }

    /// Configure progress bar for a new file
    pub fn start_file(&mut self, index: usize, path: &Path, iterations: usize) {
        let display_name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        if index >= self.file_states.len() {
            self.file_states.resize(index + 1, (String::new(), 0, 0));
        }
        if let Some(state) = self.file_states.get_mut(index) {
            *state = (display_name, 0, iterations);
        }
        self.update_bars();
    }

    /// Report current iteration and elapsed time
    pub fn update_iteration(&mut self, file_index: usize, iteration: usize, _elapsed: Duration) {
        if let Some(state) = self.file_states.get_mut(file_index) {
            state.1 = iteration;
        }
        self.update_bars();
    }

    /// Mark file as completed and update batch progress
    pub fn complete_file(&mut self, index: usize, _elapsed: Duration) {
        if let Some(ref batch_bar) = self.batch_bar {
            batch_bar.inc(1);
        }

        if let Some(state) = self.file_states.get_mut(index) {
            let max_iter = state.2;
            state.0 = format!("✓ {}", state.0);
            state.1 = max_iter;
        }
        self.update_bars();
    }

    /// Clean up all progress displays
    pub fn finish(&self) {
        if let Some(ref batch_bar) = self.batch_bar {
            batch_bar.finish_with_message("All files processed");
        }
        let _ = self.multi_progress.clear();
    }

    /// Update all progress bars to show the last N active files
    fn update_bars(&mut self) {
        // Find the last N files that are in progress or recently completed
        let mut active_files = Vec::new();
        for (i, (name, current, max)) in self.file_states.iter().enumerate() {
            // Only consider files that have started processing (current > 0) or are completed
            if !name.is_empty() && *current > 0 {
                active_files.push((i, name.clone(), *current, *max));
            }
        }

        // Take the last N files
        let start_idx = active_files
            .len()
            .saturating_sub(MAX_INDIVIDUAL_PROGRESS_BARS);
        let visible_files = active_files.get(start_idx..).unwrap_or(&[]);

        // Create progress bars dynamically as needed - only for active files
        let needed_bars = visible_files.len().min(MAX_INDIVIDUAL_PROGRESS_BARS);
        while self.file_bars.len() < needed_bars {
            let pb = ProgressBar::new(0);
            pb.set_style(Self::iteration_style());
            self.file_bars.push(self.multi_progress.add(pb));
        }

        // Update each progress bar
        for (bar_idx, (_file_idx, name, current, max)) in visible_files.iter().enumerate() {
            if let Some(bar) = self.file_bars.get(bar_idx) {
                bar.set_length(*max as u64);
                bar.set_position(*current as u64);
                let max_width = max.to_string().len();
                bar.set_message(format!("{current:>max_width$}/{max}"));
                bar.set_prefix(name.clone());
            }
        }

        // Clear any unused bars
        for bar_idx in visible_files.len()..self.file_bars.len() {
            if let Some(bar) = self.file_bars.get(bar_idx) {
                bar.set_length(0);
                bar.set_position(0);
                bar.set_message(String::new());
                bar.set_prefix(String::new());
            }
        }
    }

    fn iteration_style() -> ProgressStyle {
        PROGRESS_STYLE.clone()
    }
}

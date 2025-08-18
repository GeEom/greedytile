//! Error types and context management for algorithm operations

use std::fmt;
use std::path::PathBuf;

/// Main error type for all algorithm operations
#[derive(Debug)]
pub enum AlgorithmError {
    /// Failed to load source image from filesystem
    ImageLoad {
        /// Path to the image file
        path: PathBuf,
        /// Underlying image loading error
        source: image::ImageError,
    },

    /// Source data doesn't meet algorithm requirements
    InvalidSourceData {
        /// Description of what's wrong with the source data
        reason: String,
    },

    /// No valid position candidates during selection
    ///
    /// Occurs when all grid positions are either:
    /// - Already filled with tiles
    /// - Have zero entropy (no possible tiles)
    NoValidPositions {
        /// Algorithm iteration when this occurred
        iteration: usize,
        /// Current grid dimensions (rows, cols)
        grid_dimensions: (usize, usize),
    },

    /// Algorithm parameter validation failed
    InvalidParameter {
        /// Name of the invalid parameter
        parameter: &'static str,
        /// Provided value that failed validation
        value: String,
        /// Explanation of why the value is invalid
        reason: String,
    },

    /// Tile index exceeds available tile set
    InvalidTileIndex {
        /// The invalid tile index
        index: usize,
        /// Maximum valid tile index
        max_tiles: usize,
    },

    /// Failed to save generated image to disk
    ImageExport {
        /// Path where export was attempted
        path: PathBuf,
        /// Underlying image export error
        source: image::ImageError,
    },

    /// General file system operation failure
    FileSystem {
        /// Path involved in the operation
        path: PathBuf,
        /// Description of the operation that failed
        operation: &'static str,
        /// Underlying I/O error
        source: std::io::Error,
    },

    /// Numerical computation produced invalid result
    Computation {
        /// Name of the computation that failed
        operation: &'static str,
        /// Description of the failure
        reason: String,
    },
}

impl fmt::Display for AlgorithmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImageLoad { path, source } => {
                write!(f, "Failed to load image '{}': {source}", path.display())
            }
            Self::InvalidSourceData { reason } => {
                write!(f, "Invalid source data: {reason}")
            }
            Self::NoValidPositions {
                iteration,
                grid_dimensions,
            } => {
                write!(
                    f,
                    "No valid positions found at iteration {iteration} (grid size {}x{})",
                    grid_dimensions.0, grid_dimensions.1
                )
            }
            Self::InvalidParameter {
                parameter,
                value,
                reason,
            } => {
                write!(f, "Invalid parameter '{parameter}' = '{value}': {reason}")
            }
            Self::InvalidTileIndex { index, max_tiles } => {
                write!(f, "Tile index {index} is out of bounds (max: {max_tiles})")
            }
            Self::ImageExport { path, source } => {
                write!(
                    f,
                    "Failed to export image to '{}': {source}",
                    path.display()
                )
            }
            Self::FileSystem {
                path,
                operation,
                source,
            } => {
                write!(
                    f,
                    "File system error during {operation} on '{}': {source}",
                    path.display()
                )
            }
            Self::Computation { operation, reason } => {
                write!(f, "Computation error in {operation}: {reason}")
            }
        }
    }
}

impl std::error::Error for AlgorithmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ImageLoad { source, .. } | Self::ImageExport { source, .. } => Some(source),
            Self::FileSystem { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Convenience type alias for algorithm results
pub type Result<T> = std::result::Result<T, AlgorithmError>;

/// Additional context to enrich error messages
#[derive(Debug, Clone, Default)]
pub struct ErrorContext {
    /// Current algorithm iteration
    pub iteration: Option<usize>,
    /// World coordinates where error occurred
    pub position: Option<[i32; 2]>,
    /// Grid indices where error occurred
    pub grid_position: Option<[usize; 2]>,
    /// Operation being performed
    pub operation: Option<&'static str>,
}

/// Enriches error messages with algorithm state information
pub trait WithContext<T> {
    /// Add error context to a Result
    ///
    /// # Errors
    ///
    /// Propagates the original error with additional context applied
    fn with_context(self, context: ErrorContext) -> Result<T>;

    /// Add just the operation context
    ///
    /// # Errors
    ///
    /// Propagates the original error with the operation context applied
    fn with_operation(self, operation: &'static str) -> Result<T>;
}

impl<T, E> WithContext<T> for std::result::Result<T, E>
where
    E: Into<AlgorithmError>,
{
    fn with_context(self, context: ErrorContext) -> Result<T> {
        self.map_err(|e| {
            let mut error = e.into();
            // Only certain error types benefit from positional context
            if let AlgorithmError::NoValidPositions { iteration, .. } = &mut error
                && let Some(iter) = context.iteration
            {
                *iteration = iter;
            }
            error
        })
    }

    fn with_operation(self, operation: &'static str) -> Result<T> {
        self.with_context(ErrorContext {
            operation: Some(operation),
            ..Default::default()
        })
    }
}

impl From<image::ImageError> for AlgorithmError {
    fn from(err: image::ImageError) -> Self {
        Self::ImageLoad {
            path: PathBuf::from("<unknown>"),
            source: err,
        }
    }
}

impl From<std::io::Error> for AlgorithmError {
    fn from(err: std::io::Error) -> Self {
        Self::FileSystem {
            path: PathBuf::from("<unknown>"),
            operation: "unknown",
            source: err,
        }
    }
}

/// Create an invalid parameter error
pub fn invalid_parameter(
    parameter: &'static str,
    value: &impl ToString,
    reason: &impl ToString,
) -> AlgorithmError {
    AlgorithmError::InvalidParameter {
        parameter,
        value: value.to_string(),
        reason: reason.to_string(),
    }
}

/// Create a computation error
pub fn computation_error(operation: &'static str, reason: &impl ToString) -> AlgorithmError {
    AlgorithmError::Computation {
        operation,
        reason: reason.to_string(),
    }
}

/// Create a generic I/O error (temporary compatibility helper)
pub fn io_error(msg: &str) -> AlgorithmError {
    AlgorithmError::InvalidParameter {
        parameter: "path",
        value: String::new(),
        reason: msg.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context() {
        let result: std::result::Result<(), AlgorithmError> =
            Err(AlgorithmError::NoValidPositions {
                iteration: 0,
                grid_dimensions: (10, 10),
            });

        let context = ErrorContext {
            iteration: Some(99),
            ..Default::default()
        };

        let err = result.with_context(context).unwrap_err();
        match err {
            AlgorithmError::NoValidPositions { iteration, .. } => {
                assert_eq!(iteration, 99);
            }
            _ => unreachable!("Expected NoValidPositions error type"),
        }
    }
}

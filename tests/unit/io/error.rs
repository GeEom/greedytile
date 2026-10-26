//! Tests for error types including source chaining and message formatting

#[cfg(test)]
mod tests {
    use greedytile::AlgorithmError;
    use std::error::Error;

    // Tests error source chaining works correctly
    // Verified by breaking source chain
    #[test]
    fn test_error_source_chain() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let error = AlgorithmError::FileSystem {
            path: "/tmp/test.png".into(),
            operation: "read",
            source: io_error,
        };

        assert!(error.source().is_some());
    }

    // Tests NoValidPositions error formatting
    // Verified by omitting iteration from message
    #[test]
    fn test_no_valid_positions_error() {
        let error = AlgorithmError::NoValidPositions {
            iteration: 42,
            grid_dimensions: (10, 20),
        };

        let message = error.to_string();
        assert!(message.contains("iteration 42"));
        assert!(message.contains("10x20"));
    }

    // Tests InvalidParameter error contains all fields
    // Verified by omitting value from message
    #[test]
    fn test_invalid_parameter_error() {
        let error = AlgorithmError::InvalidParameter {
            parameter: "tile_size",
            value: "-1".to_string(),
            reason: "must be positive".to_string(),
        };

        let message = error.to_string();
        assert!(message.contains("tile_size"));
        assert!(message.contains("-1"));
        assert!(message.contains("must be positive"));
    }

    // Tests ImageExport error with IO source
    // Verified by excluding source error from message
    #[test]
    fn test_image_export_error() -> crate::TestResult {
        use std::path::PathBuf;

        let image_error = image::ImageError::IoError(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "access denied",
        ));

        let error = AlgorithmError::ImageExport {
            path: PathBuf::from("/restricted/output.png"),
            source: image_error,
        };

        let message = error.to_string();
        assert!(message.contains("/restricted/output.png"));
        assert!(error.source().is_some());

        let _source_error = error.source().ok_or("error has no source")?;
        assert!(
            message.contains("Permission denied")
                || message.contains("permission denied")
                || message.contains("access denied"),
            "Error message should include source error details: {message}"
        );
        Ok(())
    }

    // Tests Computation error formatting
    // Verified by omitting reason from message
    #[test]
    fn test_computation_error() {
        let error = AlgorithmError::Computation {
            operation: "matrix multiplication",
            reason: "dimensions mismatch".to_string(),
        };

        let message = error.to_string();
        assert!(message.contains("matrix multiplication"));
        assert!(message.contains("dimensions mismatch"));
    }

    // Tests that error context properly enriches errors with position information
    // Verified by removing position context application in with_context
    #[test]
    fn test_error_context_with_position() -> crate::TestResult {
        use greedytile::io::error::{ErrorContext, WithContext};

        let error = AlgorithmError::NoValidPositions {
            iteration: 10,
            grid_dimensions: (50, 50),
        };

        let context = ErrorContext {
            iteration: Some(42),
            position: Some([100, 200]),
            grid_position: Some([10, 20]),
            operation: Some("pattern matching"),
        };

        let enriched = std::result::Result::<(), AlgorithmError>::Err(error).with_context(context);

        let Err(AlgorithmError::NoValidPositions { iteration, .. }) = enriched else {
            return Err(format!("expected NoValidPositions, got {enriched:?}").into());
        };
        assert_eq!(iteration, 42);
        Ok(())
    }

    // Tests helper functions create properly formatted errors
    // Verified by changing helper function implementations
    #[test]
    fn test_error_helper_functions() -> crate::TestResult {
        use greedytile::io::error::{computation_error, invalid_parameter};

        let param_err = invalid_parameter("width", &-5, &"must be positive");
        let AlgorithmError::InvalidParameter {
            parameter,
            value,
            reason,
        } = param_err
        else {
            return Err(format!("expected InvalidParameter, got {param_err:?}").into());
        };
        assert_eq!(parameter, "width");
        assert_eq!(value, "-5");
        assert_eq!(reason, "must be positive");

        let comp_err = computation_error("entropy calculation", &"division by zero");
        let AlgorithmError::Computation {
            operation,
            reason: comp_reason,
        } = comp_err
        else {
            return Err(format!("expected Computation, got {comp_err:?}").into());
        };
        assert_eq!(operation, "entropy calculation");
        assert_eq!(comp_reason, "division by zero");
        Ok(())
    }
}

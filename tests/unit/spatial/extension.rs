//! Tests for dynamic grid extension including padding calculations and array copying

#[cfg(test)]
mod tests {
    use crate::spatial::extension::{
        Extendable, ExtensionInfo, calculate_extension, extend_array_2d, extend_array_3d,
    };
    use ndarray::{Array2, Array3};

    // Tests padding values for f64 (1.0) and u32 (1)
    // Verified by changing padding values
    #[test]
    fn test_extendable_trait_implementations() {
        assert!((f64::padding_value() - 1.0).abs() < f64::EPSILON);
        assert_eq!(u32::padding_value(), 1);
    }

    // Tests when coordinates are within bounds
    // Verified by forcing needs_extension to true
    #[test]
    fn test_calculate_extension_no_extension_needed() {
        let current_dims = [10, 10];
        let offset = [5, 5];
        let coordinates = [2, 2];
        let radius = 2;

        let info = calculate_extension(current_dims, offset, coordinates, radius);

        assert!(!info.needs_extension);
        assert_eq!(info.pad_left, 0);
        assert_eq!(info.pad_right, 0);
        assert_eq!(info.pad_top, 0);
        assert_eq!(info.pad_bottom, 0);
        assert_eq!(info.new_offset, offset);
    }

    // Tests padding calculation when expanding
    // Verified by halving padding calculations
    #[test]
    fn test_calculate_extension_needs_expansion() {
        let current_dims = [5, 5];
        let offset = [2, 2];
        let coordinates = [8, 8];
        let radius = 2;

        let info = calculate_extension(current_dims, offset, coordinates, radius);

        assert!(info.needs_extension);
        assert_eq!(info.pad_left, 0);
        assert_eq!(info.pad_right, 8);
        assert_eq!(info.pad_top, 0);
        assert_eq!(info.pad_bottom, 8);
        assert_eq!(info.new_offset, [2, 2]);
    }

    // Tests offset adjustment for negative coordinates
    // Verified by removing offset adjustment
    #[test]
    fn test_calculate_extension_negative_padding() {
        let current_dims = [5, 5];
        let offset = [2, 2];
        let coordinates = [-3, -3];
        let radius = 2;

        let info = calculate_extension(current_dims, offset, coordinates, radius);

        assert!(info.needs_extension);
        assert_eq!(info.pad_left, 3);
        assert_eq!(info.pad_right, 0);
        assert_eq!(info.pad_top, 3);
        assert_eq!(info.pad_bottom, 0);
        assert_eq!(info.new_offset, [5, 5]);
    }

    // Tests 2D array with no extension needed
    // Verified by removing early return optimization
    #[test]
    fn test_extend_array_2d_no_extension() -> crate::TestResult {
        let mut array = Array2::from_elem((3, 3), 5.0);
        if let Some(elem) = array.get_mut([0, 0]) {
            *elem = 1.0;
        }
        if let Some(elem) = array.get_mut([1, 1]) {
            *elem = 2.0;
        }
        if let Some(elem) = array.get_mut([2, 2]) {
            *elem = 3.0;
        }

        let info = ExtensionInfo {
            pad_left: 0,
            pad_right: 0,
            pad_top: 0,
            pad_bottom: 0,
            new_offset: [0, 0],
            needs_extension: false,
        };

        let extended = extend_array_2d(&array, &info, 1.0);

        assert_eq!(extended.dim(), (3, 3));

        assert!(
            (*extended.get([0, 0]).ok_or("cell outside array")? - 1.0_f64).abs() < f64::EPSILON
        );
        assert!(
            (*extended.get([1, 1]).ok_or("cell outside array")? - 2.0_f64).abs() < f64::EPSILON
        );
        assert!(
            (*extended.get([2, 2]).ok_or("cell outside array")? - 3.0_f64).abs() < f64::EPSILON
        );
        assert!(
            (*extended.get([0, 1]).ok_or("cell outside array")? - 5.0_f64).abs() < f64::EPSILON
        );

        assert_eq!(extended, array);
        Ok(())
    }

    // Tests 2D array padding with values preserved
    // Verified by omitting value copy loop
    #[test]
    fn test_extend_array_2d_with_padding() {
        let mut array = Array2::ones((3, 3));
        if let Some(elem) = array.get_mut([1, 1]) {
            *elem = 5.0;
        }

        let info = ExtensionInfo {
            pad_left: 1,
            pad_right: 2,
            pad_top: 1,
            pad_bottom: 2,
            new_offset: [1, 1],
            needs_extension: true,
        };

        let extended = extend_array_2d(&array, &info, 0.0);

        assert_eq!(extended.dim(), (6, 6));
        assert!(
            extended
                .get([2, 2])
                .is_some_and(|&v: &f64| (v - 5.0).abs() < f64::EPSILON)
        );
        assert!(
            extended
                .get([0, 0])
                .is_some_and(|&v: &f64| v.abs() < f64::EPSILON)
        );
        assert!(
            extended
                .get([5, 5])
                .is_some_and(|&v: &f64| v.abs() < f64::EPSILON)
        );
    }

    // Tests 3D array with no extension needed
    // Verified by removing early return optimization
    #[test]
    fn test_extend_array_3d_no_extension() -> crate::TestResult {
        let mut array = Array3::from_elem((2, 3, 3), 5.0);
        if let Some(elem) = array.get_mut([0, 0, 0]) {
            *elem = 1.0;
        }
        if let Some(elem) = array.get_mut([0, 1, 1]) {
            *elem = 2.0;
        }
        if let Some(elem) = array.get_mut([1, 2, 2]) {
            *elem = 3.0;
        }
        if let Some(elem) = array.get_mut([1, 0, 0]) {
            *elem = 4.0;
        }

        let info = ExtensionInfo {
            pad_left: 0,
            pad_right: 0,
            pad_top: 0,
            pad_bottom: 0,
            new_offset: [0, 0],
            needs_extension: false,
        };

        let extended = extend_array_3d(&array, &info);

        assert_eq!(extended.dim(), (2, 3, 3));

        assert!(
            (*extended.get([0, 0, 0]).ok_or("cell outside array")? - 1.0_f64).abs() < f64::EPSILON
        );
        assert!(
            (*extended.get([0, 1, 1]).ok_or("cell outside array")? - 2.0_f64).abs() < f64::EPSILON
        );
        assert!(
            (*extended.get([1, 2, 2]).ok_or("cell outside array")? - 3.0_f64).abs() < f64::EPSILON
        );
        assert!(
            (*extended.get([1, 0, 0]).ok_or("cell outside array")? - 4.0_f64).abs() < f64::EPSILON
        );
        assert!(
            (*extended.get([0, 0, 1]).ok_or("cell outside array")? - 5.0_f64).abs() < f64::EPSILON
        );

        assert_eq!(extended, array);
        Ok(())
    }

    // Tests 3D array padding with values preserved
    // Verified by omitting value copy loop
    #[test]
    fn test_extend_array_3d_with_padding() {
        let mut array = Array3::ones((2, 3, 3));
        if let Some(elem) = array.get_mut([0, 1, 1]) {
            *elem = 5.0;
        }
        if let Some(elem) = array.get_mut([1, 2, 2]) {
            *elem = 7.0;
        }

        let info = ExtensionInfo {
            pad_left: 1,
            pad_right: 2,
            pad_top: 1,
            pad_bottom: 2,
            new_offset: [1, 1],
            needs_extension: true,
        };

        let extended = extend_array_3d(&array, &info);

        assert_eq!(extended.dim(), (2, 6, 6));
        assert!(
            extended
                .get([0, 2, 2])
                .is_some_and(|&v: &f64| (v - 5.0).abs() < f64::EPSILON)
        );
        assert!(
            extended
                .get([1, 3, 3])
                .is_some_and(|&v: &f64| (v - 7.0).abs() < f64::EPSILON)
        );
        assert!(
            extended
                .get([0, 0, 0])
                .is_some_and(|&v: &f64| (v - 1.0).abs() < f64::EPSILON)
        );
        assert!(
            extended
                .get([1, 5, 5])
                .is_some_and(|&v: &f64| (v - 1.0).abs() < f64::EPSILON)
        );
    }

    // Tests ExtensionInfo is Copy and Clone
    // Verified by removing Copy trait
    #[test]
    fn test_extension_info_copy_clone() {
        let info = ExtensionInfo {
            pad_left: 1,
            pad_right: 2,
            pad_top: 3,
            pad_bottom: 4,
            new_offset: [5, 6],
            needs_extension: true,
        };

        let cloned = info;
        let copied = info;

        assert_eq!(cloned.pad_left, 1);
        assert_eq!(copied.pad_left, 1);
    }
}

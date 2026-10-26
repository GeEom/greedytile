//! Tests for prefill image parsing and queue management

use crate::io::prefill::{PrefillData, PrefillPlacement};

#[cfg(test)]
mod tests {
    use super::*;

    // Tests PrefillPlacement struct creation
    // Verified by removing Clone derive
    #[test]
    fn test_prefill_placement_creation() {
        let test_cases = vec![
            ([0, 0], 1),
            ([i32::MAX, i32::MIN], usize::MAX),
            ([-100, 200], 42),
        ];

        for (world_pos, tile_ref) in test_cases {
            let placement = PrefillPlacement {
                world_position: world_pos,
                tile_reference: tile_ref,
            };

            assert_eq!(placement.world_position, world_pos);
            assert_eq!(placement.tile_reference, tile_ref);
        }

        let original = PrefillPlacement {
            world_position: [10, 20],
            tile_reference: 5,
        };

        let cloned = original.clone();
        assert_eq!(cloned.world_position, original.world_position);
        assert_eq!(cloned.tile_reference, original.tile_reference);
    }

    // Tests protected position checking
    // Verified by breaking is_protected to return None
    #[test]
    fn test_prefill_data_is_protected() {
        let placement_queue = std::collections::VecDeque::new();
        let mut protected_positions = std::collections::BTreeMap::new();

        protected_positions.insert([5, 10], 2);
        protected_positions.insert([-3, 7], 4);

        let bounds = crate::spatial::grid::BoundingBox {
            min: [-3, 7],
            max: [5, 10],
        };

        let prefill_data = PrefillData {
            placement_queue,
            protected_positions,
            bounds,
        };

        assert_eq!(prefill_data.is_protected([5, 10]), Some(2));
        assert_eq!(prefill_data.is_protected([-3, 7]), Some(4));
        assert_eq!(prefill_data.is_protected([0, 0]), None);
    }

    // Tests next_placement and queue_replacement
    // Verified by changing queue_replacement to push_back
    #[test]
    fn test_prefill_data_queue_operations() -> crate::TestResult {
        let mut placement_queue = std::collections::VecDeque::new();
        let placement1 = PrefillPlacement {
            world_position: [1, 2],
            tile_reference: 3,
        };
        let placement2 = PrefillPlacement {
            world_position: [4, 5],
            tile_reference: 6,
        };

        placement_queue.push_back(placement1);
        placement_queue.push_back(placement2);

        let bounds = crate::spatial::grid::BoundingBox {
            min: [1, 2],
            max: [4, 5],
        };

        let mut prefill_data = PrefillData {
            placement_queue,
            protected_positions: std::collections::BTreeMap::new(),
            bounds,
        };

        let next = prefill_data.next_placement();
        assert!(next.is_some());
        let next = next.ok_or("queue is empty")?;
        assert_eq!(next.world_position, [1, 2]);
        assert_eq!(next.tile_reference, 3);

        let replacement = PrefillPlacement {
            world_position: [7, 8],
            tile_reference: 9,
        };
        prefill_data.queue_replacement(replacement);

        let replacement_next = prefill_data.next_placement();
        assert!(replacement_next.is_some());
        let replacement_next = replacement_next.ok_or("queue is empty")?;
        assert_eq!(replacement_next.world_position, [7, 8]);
        assert_eq!(replacement_next.tile_reference, 9);
        Ok(())
    }
}

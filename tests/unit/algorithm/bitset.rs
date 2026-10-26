//! Tests for `TileBitset` operations including set operations and conversions

#[cfg(test)]
mod tests {
    use greedytile::algorithm::bitset::TileBitset;
    use std::collections::BTreeSet;

    // Verifies new TileBitset is empty with count 0
    // Verified by initializing bitset with all bits set to 1
    #[test]
    fn test_new_bitset() {
        let bitset = TileBitset::new(10);
        assert_eq!(bitset.count(), 0);
        assert!(bitset.is_empty());
    }

    // Tests insertion and containment checking
    // Verified by removing the bit-setting logic from insert method
    #[test]
    fn test_insert_and_contains() {
        let mut bitset = TileBitset::new(10);
        bitset.insert(5);
        assert!(bitset.contains(5));
        assert!(!bitset.contains(3));
        assert_eq!(bitset.count(), 1);
    }

    // Tests intersection of two bitsets returns correct elements
    // Verified by changing intersection operation to union operation
    #[test]
    fn test_intersection() {
        let mut set1 = TileBitset::new(10);
        set1.insert(1);
        set1.insert(3);
        set1.insert(5);

        let mut set2 = TileBitset::new(10);
        set2.insert(3);
        set2.insert(5);
        set2.insert(7);

        let intersection = set1.intersection(&set2);
        let result = intersection.to_vec();
        assert_eq!(result, vec![3, 5]);
    }

    // Tests conversion from BTreeSet to TileBitset
    // Verified by removing the element insertion loop
    #[test]
    fn test_from_set() {
        let mut hashset = BTreeSet::new();
        hashset.insert(1);
        hashset.insert(3);
        hashset.insert(5);

        let bitset = TileBitset::from_set(&hashset, 10);
        assert!(bitset.contains(1));
        assert!(bitset.contains(3));
        assert!(bitset.contains(5));
        assert!(!bitset.contains(2));
        assert_eq!(bitset.count(), 3);
    }

    // Tests creation of bitset with all bits set
    // Verified by initializing all bits to 0 instead of 1
    #[test]
    fn test_all_bits_set() {
        let bitset = TileBitset::all(5);
        for i in 1..=5 {
            assert!(bitset.contains(i));
        }
        assert_eq!(bitset.count(), 5);
    }
}

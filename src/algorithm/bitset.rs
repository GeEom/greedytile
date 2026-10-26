use bitvec::bitvec;
use bitvec::vec::BitVec;
use std::collections::BTreeSet;
use std::fmt;

/// A fixed-size set of tile references, which are 1-based like everywhere else
#[derive(Clone, Debug)]
pub struct TileBitset {
    bits: BitVec,
    max_tiles: usize,
}

impl TileBitset {
    /// Create a bitset with no tiles present
    #[must_use]
    pub fn new(max_tiles: usize) -> Self {
        Self {
            bits: bitvec![0; max_tiles],
            max_tiles,
        }
    }

    /// Create a bitset containing all possible tiles
    #[must_use]
    pub fn all(max_tiles: usize) -> Self {
        Self {
            bits: bitvec![1; max_tiles],
            max_tiles,
        }
    }

    /// Insert a tile reference; 0 and those past the size are ignored
    pub fn insert(&mut self, tile: usize) {
        if tile > 0 && tile <= self.max_tiles {
            self.bits.set(tile - 1, true);
        }
    }

    /// Test tile membership
    #[must_use]
    pub fn contains(&self, tile: usize) -> bool {
        if tile > 0 {
            self.bits.get(tile - 1).as_deref() == Some(&true)
        } else {
            false
        }
    }

    /// Intersect this bitset with another in-place
    pub fn intersect_with(&mut self, other: &Self) {
        self.bits &= &other.bits;
    }

    /// Create a new bitset containing the intersection
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.intersect_with(other);
        result
    }

    /// Test if no tiles are present
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bits.not_any()
    }

    /// Count tiles in the set
    #[must_use]
    pub fn count(&self) -> usize {
        self.bits.count_ones()
    }

    /// The tile references in the set, ascending
    #[must_use]
    pub fn to_vec(&self) -> Vec<usize> {
        self.bits.iter_ones().map(|index| index + 1).collect()
    }

    /// Convert from a set of 1-based tile indices
    #[must_use]
    pub fn from_set(set: &BTreeSet<usize>, max_tiles: usize) -> Self {
        let mut bitset = Self::new(max_tiles);
        for &tile in set {
            bitset.insert(tile);
        }
        bitset
    }
}

impl fmt::Display for TileBitset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TileBitset({} tiles: {:?})", self.count(), self.to_vec())
    }
}

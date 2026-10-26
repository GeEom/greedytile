use crate::algorithm::bitset::TileBitset;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

/// A 3x3 neighbourhood and the cell in it whose viable values are cached
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PatternKey {
    pattern: [[i32; 3]; 3],
    target_row: usize,
    target_col: usize,
}

impl PatternKey {
    /// Create a pattern key from the surrounding tile pattern
    #[must_use]
    pub const fn new(tile_pattern: &[[i32; 3]; 3], target_row: usize, target_col: usize) -> Self {
        Self {
            pattern: *tile_pattern,
            target_row,
            target_col,
        }
    }
}

/// Viable values per neighbourhood, so repeated neighbourhoods skip the matching
#[derive(Default)]
pub struct ViableTilesCache {
    /// Pattern to viable tiles mapping
    pattern_cache: BTreeMap<PatternKey, TileBitset>,

    /// Cache performance statistics
    pub stats: CacheStats,
}

/// Performance metrics for cache effectiveness
#[derive(Default, Debug)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
}

impl ViableTilesCache {
    /// Create an empty cache
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The cached set for `pattern_key`, computed by `compute_fn` on a miss
    pub fn get_or_compute_pattern<F>(
        &mut self,
        pattern_key: PatternKey,
        compute_fn: F,
    ) -> &TileBitset
    where
        F: FnOnce() -> TileBitset,
    {
        match self.pattern_cache.entry(pattern_key) {
            Entry::Occupied(entry) => {
                self.stats.hits += 1;
                entry.into_mut()
            }
            Entry::Vacant(entry) => {
                self.stats.misses += 1;
                entry.insert(compute_fn())
            }
        }
    }
}

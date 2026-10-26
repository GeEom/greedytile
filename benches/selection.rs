//! Performance measurement for tile viability computation at varying grid densities

use criterion::{BenchmarkId, Criterion};
use greedytile::algorithm::cache::ViableTilesCache;
use greedytile::algorithm::executor::GreedyStochastic;
use greedytile::algorithm::selection::compute_viable_tiles_at_position;
use std::hint::black_box;

/// Measures viability computation cost as grid density increases from 0% to 75%
fn bench_compute_viable_tiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_viable_tiles");

    for fill_percent in &[0, 25, 50, 75] {
        let Ok(mut executor) = GreedyStochastic::new(12345) else {
            group.finish();
            return;
        };

        let target_fill = (fill_percent * 250) / 100;
        for _ in 0..target_fill {
            if executor.run_iteration().is_err() {
                group.finish();
                return;
            }
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(fill_percent),
            fill_percent,
            |b, _| {
                b.iter(|| {
                    let mut cache = ViableTilesCache::new();
                    let positions = vec![[10, 10], [15, 15], [20, 20], [25, 25], [30, 30]];

                    for pos in &positions {
                        let viable = compute_viable_tiles_at_position(
                            &executor.grid_state,
                            black_box(*pos),
                            executor.system_offset,
                            &executor.step_data.source_tiles,
                            &executor.step_data,
                            &mut cache,
                        );
                        black_box(viable);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Measures single viability computation with 40% grid fill
fn bench_compute_viable_tiles_single_position(c: &mut Criterion) {
    let Ok(mut executor) = GreedyStochastic::new(12345) else {
        return;
    };

    for _ in 0..100 {
        if executor.run_iteration().is_err() {
            return;
        }
    }

    c.bench_function("compute_viable_tiles_single_call", |b| {
        b.iter(|| {
            let mut cache = ViableTilesCache::new();
            compute_viable_tiles_at_position(
                &executor.grid_state,
                black_box([10, 10]),
                executor.system_offset,
                &executor.step_data.source_tiles,
                &executor.step_data,
                &mut cache,
            )
        });
    });
}

fn main() {
    let mut criterion = Criterion::default().configure_from_args();
    bench_compute_viable_tiles(&mut criterion);
    bench_compute_viable_tiles_single_position(&mut criterion);
    criterion.final_summary();
}

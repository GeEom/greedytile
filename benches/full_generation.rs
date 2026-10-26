//! Performance measurement for complete pattern generation workflow

use criterion::Criterion;
use greedytile::algorithm::executor::GreedyStochastic;
use std::hint::black_box;

/// Measures time to complete 250 algorithm iterations including grid expansion
fn bench_generate_250_steps(c: &mut Criterion) {
    c.bench_function("generate_250_steps", |b| {
        b.iter(|| {
            let Ok(mut executor) = GreedyStochastic::new(12345) else {
                return;
            };

            for _ in 0..250 {
                if executor.run_iteration().is_err() {
                    return;
                }
            }
            black_box(executor.iteration);
        });
    });
}

fn main() {
    let mut criterion = Criterion::default().configure_from_args();
    bench_generate_250_steps(&mut criterion);
    criterion.final_summary();
}

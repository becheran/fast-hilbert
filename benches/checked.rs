use core::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};

#[allow(clippy::too_many_lines)]
fn criterion_benchmark(c: &mut Criterion) {
    let bits: usize = 8;
    let n: usize = 2usize.pow(bits as u32);
    c.bench_function("fast_hilbert_checked", |b| {
        b.iter(|| {
            for x in 0..n {
                for y in 0..n {
                    black_box(
                        fast_hilbert::xy2h_checked(
                            black_box(x as u32),
                            black_box(y as u32),
                            black_box(bits as u8),
                        )
                        .unwrap(),
                    );
                }
            }
        });
    });

    let xy_low: (u32, u32) = (1, 2);
    let xy_high: (u32, u32) = (u32::MAX - 1, u32::MAX - 2);
    let order: u8 = 32;
    c.bench_function("fast_hilbert_low_checked", |b| {
        b.iter(|| {
            black_box(
                fast_hilbert::xy2h_checked(
                    black_box(xy_low.0),
                    black_box(xy_low.1),
                    black_box(order),
                )
                .unwrap(),
            );
        });
    });
    c.bench_function("fast_hilbert_high_checked", |b| {
        b.iter(|| {
            black_box(
                fast_hilbert::xy2h_checked(
                    black_box(xy_high.0),
                    black_box(xy_high.1),
                    black_box(order),
                )
                .unwrap(),
            );
        });
    });
}
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(2000);
    targets = criterion_benchmark
);
criterion_main!(benches);

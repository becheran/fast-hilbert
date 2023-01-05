use core::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    let bits: usize = 8;
    let n: usize = 2usize.pow(bits as u32);

    c.bench_function("hilbert_curve", |b| {
        b.iter(|| {
            for x in 0..n {
                for y in 0..n {
                    black_box(hilbert_curve::convert_2d_to_1d(
                        black_box(x),
                        black_box(y),
                        black_box(n),
                    ));
                }
            }
        })
    });

    c.bench_function("hilbert_2d", |b| {
        b.iter(|| {
            for x in 0..n {
                for y in 0..n {
                    black_box(hilbert_2d::xy2h_discrete(
                        black_box(x),
                        black_box(y),
                        black_box(bits),
                        black_box(hilbert_2d::Variant::Hilbert),
                    ));
                }
            }
        })
    });

    c.bench_function("hilbert", |b| {
        b.iter(|| {
            for x in 0..n {
                for y in 0..n {
                    let p = hilbert::Point::new(0, &[black_box(x as u32), black_box(y as u32)]);
                    black_box(p.hilbert_transform(black_box(bits)));
                }
            }
        })
    });

    c.bench_function("fast_hilbert", |b| {
        b.iter(|| {
            for x in 0..n {
                for y in 0..n {
                    black_box(fast_hilbert::xy2h(
                        black_box(x as u32),
                        black_box(y as u32),
                        black_box(bits as u8),
                    ));
                }
            }
        })
    });

    let xy_low: (u32, u32) = (1, 2);
    let xy_high: (u32, u32) = (u32::MAX - 1, u32::MAX - 2);
    let order: u8 = 32;
    let n: usize = 2usize.pow(order as u32);
    c.bench_function("fast_hilbert_low", |b| {
        b.iter(|| {
            black_box(fast_hilbert::xy2h(black_box(xy_low.0), black_box(xy_low.1), black_box(order)));
        })
    });
    c.bench_function("fast_hilbert_high", |b| {
        b.iter(|| {
            black_box(fast_hilbert::xy2h(black_box(xy_high.0), black_box(xy_high.1), black_box(order)));
        })
    });
    c.bench_function("hilbert_curve_low", |b| {
        b.iter(|| {
            black_box(hilbert_curve::convert_2d_to_1d(xy_low.0 as usize, xy_low.1 as usize, n));
        })
    });
    c.bench_function("hilbert_curve_high", |b| {
        b.iter(|| {
            black_box(hilbert_curve::convert_2d_to_1d(xy_high.0 as usize, xy_high.1 as usize, n));
        })
    });
    c.bench_function("hilbert_2d_low", |b| {
        b.iter(|| {
            black_box(hilbert_2d::xy2h_discrete(
                xy_low.0 as usize,
                xy_low.1 as usize,
                order as usize,
                hilbert_2d::Variant::Hilbert,
            ));
        })
    });
    c.bench_function("hilbert_2d_high", |b| {
        b.iter(|| {
            black_box(hilbert_2d::xy2h_discrete(
                xy_high.0 as usize,
                xy_high.1 as usize,
                order as usize,
                hilbert_2d::Variant::Hilbert,
            ));
        })
    });
    c.bench_function("hilbert_low", |b| {
        b.iter(|| {
            let p = hilbert::Point::new(0, &[xy_low.0, xy_low.1]);
            black_box(p.hilbert_transform(order as usize));
        })
    });
    c.bench_function("hilbert_high", |b| {
        b.iter(|| {
            let p = hilbert::Point::new(0, &[xy_high.0, xy_high.1]);
            black_box(p.hilbert_transform(order as usize));
        })
    });
}
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(2000);
    targets = criterion_benchmark
);
criterion_main!(benches);

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    let bits: usize = 8;
    let n: usize = 2usize.pow(bits as u32);

    c.bench_function("hilbert_curve", |b| {
        b.iter(|| {
            for x in 0..n {
                for y in 0..n {
                    hilbert_curve::convert_2d_to_1d(black_box(x), black_box(y), black_box(n));
                }
            }
        })
    });

    c.bench_function("hilbert_2d", |b| {
        b.iter(|| {
            for x in 0..n {
                for y in 0..n {
                    hilbert_2d::xy2h_discrete(
                        black_box(x),
                        black_box(y),
                        black_box(bits),
                        black_box(hilbert_2d::Variant::Hilbert),
                    );
                }
            }
        })
    });

    c.bench_function("hilbert", |b| {
        b.iter(|| {
            for x in 0..n {
                for y in 0..n {
                    let p = hilbert::Point::new(0, &[black_box(x as u32), black_box(y as u32)]);
                    p.hilbert_transform(black_box(bits));
                }
            }
        })
    });

    c.bench_function("fast_hilbert", |b| {
        b.iter(|| {
            for x in 0..n {
                for y in 0..n {
                    fast_hilbert::xy2h(black_box(x as u32), black_box(y as u32),black_box(bits as u8));
                }
            }
        })
    });
}
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(2000);
    targets = criterion_benchmark
);
criterion_main!(benches);

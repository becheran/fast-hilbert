use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    let bits: usize = 32;
    let n: usize = 2usize.pow(bits as u32);

    c.bench_function("hilbert_curve", |b| {
        b.iter(|| {
            hilbert_curve::convert_2d_to_1d(black_box(42), black_box(42), black_box(n));
        })
    });

    c.bench_function("hilbert_2d", |b| {
        b.iter(|| {
            hilbert_2d::xy2h_discrete(
                black_box(42),
                black_box(42),
                black_box(bits),
                black_box(hilbert_2d::Variant::Hilbert),
            );
        })
    });

    c.bench_function("hilbert", |b| {
        b.iter(|| {
            let p = hilbert::Point::new(0, &[black_box(42 as u32), black_box(42 as u32)]);
            p.hilbert_transform(black_box(bits))
        })
    });

    c.bench_function("fast_hilbert", |b| {
        b.iter(|| {
            fast_hilbert::xy2h(
                black_box(42 as u32),
                black_box(42 as u32),
                black_box(bits as u8),
            );
        })
    });

    c.bench_function("lindel", |b| {
        b.iter(|| {
            lindel::hilbert_encode(black_box([42u32, 42]));
        })
    });
}
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(2000);
    targets = criterion_benchmark
);
criterion_main!(benches);

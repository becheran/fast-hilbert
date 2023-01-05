use core::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    let bits: usize = 8;
    let n: usize = 2usize.pow(bits as u32);

    c.bench_function("hilbert_cpp", |b| {
        let n_u16 = n as u16;
        b.iter(|| {
            for x in 0..n_u16 {
                for y in 0..n_u16 {
                    black_box(hilbert_xy_to_index(black_box(x), black_box(y)));
                }
            }
        })
    });
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

fn hilbert_xy_to_index(x: u16, y: u16) -> u32 {
    let x = x as u32;
    let y = y as u32;

    // Fast Hilbert curve algorithm by http://threadlocalmutex.com/
    // Ported from C++ https://github.com/rawrunprotected/hilbert_curves (public domain)
    let mut a_1 = x ^ y;
    let mut b_1 = 0xFFFF ^ a_1;
    let mut c_1 = 0xFFFF ^ (x | y);
    let mut d_1 = x & (y ^ 0xFFFF);

    let mut a_2 = a_1 | (b_1 >> 1);
    let mut b_2 = (a_1 >> 1) ^ a_1;
    let mut c_2 = ((c_1 >> 1) ^ (b_1 & (d_1 >> 1))) ^ c_1;
    let mut d_2 = ((a_1 & (c_1 >> 1)) ^ (d_1 >> 1)) ^ d_1;

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    a_2 = (a_1 & (a_1 >> 2)) ^ (b_1 & (b_1 >> 2));
    b_2 = (a_1 & (b_1 >> 2)) ^ (b_1 & ((a_1 ^ b_1) >> 2));
    c_2 ^= (a_1 & (c_1 >> 2)) ^ (b_1 & (d_1 >> 2));
    d_2 ^= (b_1 & (c_1 >> 2)) ^ ((a_1 ^ b_1) & (d_1 >> 2));

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    a_2 = (a_1 & (a_1 >> 4)) ^ (b_1 & (b_1 >> 4));
    b_2 = (a_1 & (b_1 >> 4)) ^ (b_1 & ((a_1 ^ b_1) >> 4));
    c_2 ^= (a_1 & (c_1 >> 4)) ^ (b_1 & (d_1 >> 4));
    d_2 ^= (b_1 & (c_1 >> 4)) ^ ((a_1 ^ b_1) & (d_1 >> 4));

    a_1 = a_2;
    b_1 = b_2;
    c_1 = c_2;
    d_1 = d_2;
    c_2 ^= (a_1 & (c_1 >> 8)) ^ (b_1 & (d_1 >> 8));
    d_2 ^= (b_1 & (c_1 >> 8)) ^ ((a_1 ^ b_1) & (d_1 >> 8));

    a_1 = c_2 ^ (c_2 >> 1);
    b_1 = d_2 ^ (d_2 >> 1);

    let mut i0 = x ^ y;
    let mut i1 = b_1 | (0xFFFF ^ (i0 | a_1));

    i0 = (i0 | (i0 << 8)) & 0x00FF00FF;
    i0 = (i0 | (i0 << 4)) & 0x0F0F0F0F;
    i0 = (i0 | (i0 << 2)) & 0x33333333;
    i0 = (i0 | (i0 << 1)) & 0x55555555;

    i1 = (i1 | (i1 << 8)) & 0x00FF00FF;
    i1 = (i1 | (i1 << 4)) & 0x0F0F0F0F;
    i1 = (i1 | (i1 << 2)) & 0x33333333;
    i1 = (i1 | (i1 << 1)) & 0x55555555;

    (i1 << 1) | i0
}
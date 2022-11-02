# Fast Hilbert

[![Build Status](https://github.com/becheran/fast-hilbert/workflows/Test/badge.svg)](https://github.com/becheran/fast-hilbert/actions?workflow=Test)
[![doc](https://docs.rs/fast_hilbert/badge.svg)](https://docs.rs/fast_hilbert)
[![crates.io](https://img.shields.io/crates/v/fast_hilbert.svg)](https://crates.io/crates/fast_hilbert)
[![usage](https://badgen.net/crates/d/fast_hilbert)](https://crates.io/crates/fast_hilbert)
[![license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fast Hilbert 2D curve computation using an efficient *Lookup Table (LUT)* and only considering the lowest order for a given input.

![h1](./doc/h1.png)
![h2](./doc/h2.png)
![h3](./doc/h3.png)
![h4](./doc/h4.png)
![h5](./doc/h5.png)
![h6](./doc/h6.png)

* Convert from discrete 2D space to 1D hilbert space and reverse
* Generalized for different unsigned integer input types (thanks [DoubleHyphen](https://github.com/DoubleHyphen) [PR#3](https://github.com/becheran/fast-hilbert/pull/3))
* Speedup via lowest order computation (thanks [DoubleHyphen](https://github.com/DoubleHyphen) [PR#2](https://github.com/becheran/fast-hilbert/pull/2))
* Very fast using an efficient 512 Byte *LUT*
* Only one additional [dependency](https://crates.io/crates/num-traits)

Benchmarking the conversion from full 256x256 discrete 2D space to the 1D hilbert space, shows that *fast_hilbert* is about **12 times faster** compared to the fastest 2D hilbert transformation libs written in rust. Benchmarked on a *Intel i5-6400 CPU @ 2.70 GHz, 4 Cores* with *8 GB RAM*:

| Library                                                | Time        | Description       |
 ------------------------------------------------------- |------------:| ----------------- |
| **fast_hilbert**                                       |  **0.08 ms** | Optimized for fast computation in 2D discrete space using an efficient *LUT*
| [hilbert_2d](https://crates.io/crates/hilbert_2d)      |  2.5 ms     | Also allows other variants such as *Moore* and *LIU* |
| [hilbert_curve](https://crates.io/crates/hilbert_curve)|   2.0 ms    | Implements algorithm described on [Wikipedia](https://en.wikipedia.org/wiki/Hilbert_curve) |
| [hilbert](https://crates.io/crates/hilbert)            |  32.1 ms    | Allows computation of higher dimensional Hilbert curves |

Especially for higher orders **fast_hilbert** outperforms other libraries by using only the next lowest relevant order instead of computing the hilbert curve bit per bit for the given input. See PR [#2](https://github.com/becheran/fast-hilbert/pull/2) and [#9](https://github.com/becheran/fast-hilbert/pull/9) for more details.

For example the computation of `xy2h(1, 2, 64)` is very fast to compute using `fast_hilbert` compared to a higher x,y pair such as `xy2h(u32::MAX-1, u32::MAX-2, 64)`:

| Library          | x=1, y=2, order=64  | x=u32::MAX-1, y=u32::MAX-2, order=64    |
 ----------------- | ------------------: | --------------------------------------: |
| **fast_hilbert** |  **1.1 ns**         | **1.2 ns**                              |
| hilbert_2d       |  73 ns              | 72 ns                                   |
| hilbert_curve    |  67 ns              | 49 ns                                   |
| hilbert          |  690 ns             | 680 ns                                  |

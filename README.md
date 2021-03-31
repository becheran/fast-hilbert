# Fast Hilbert

[![Build Status](https://github.com/becheran/fast-hilbert/workflows/Test/badge.svg)](https://github.com/becheran/fast-hilbert/actions?workflow=Test)
[![doc](https://docs.rs/fast_hilbert/badge.svg)](https://docs.rs/fast_hilbert)
[![crates.io](https://img.shields.io/crates/v/fast_hilbert.svg)](https://crates.io/crates/fast_hilbert)
[![usage](https://badgen.net/crates/d/fast_hilbert)](https://crates.io/crates/fast_hilbert)
[![license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fast Hilbert 2D curve computation using an efficient *Lookup Table (LUT)* and a more efficient "orientation-stable encoding". The curve is slightly than the original hilbert curve. Every odd iteration is oriented by 90 degrees compared to the original hilbert implementation.

![h1](./doc/h1.png)
![h2](./doc/h2.png)
![h3](./doc/h3.png)
![h4](./doc/h4.png)
![h5](./doc/h5.png)
![h6](./doc/h6.png)

* *Orientation-stable encoding* (all credits to [DoubleHyphen](https://github.com/DoubleHyphen) see [this PR](https://github.com/becheran/fast-hilbert/pull/2) for more information)
* Convert from discrete 2D space to 1D hilbert space and reverse
* No `order` or `iteration` input required
* Very fast using an efficient 512 Byte *LUT*
* Only one additional [dependency](https://crates.io/crates/num-traits).

Benchmarking the conversion from full 256x256 discrete 2D space to the 1D hilbert space, shows that *fast_hilbert* is about **12 times faster** compared to the fastest 2D hilbert transformation libs written in rust. Benchmarked on a *Intel i5-6400 CPU @ 2.70 GHz, 4 Cores* with *8 GB RAM*:

| Library          | Time       | Description       |
 ----------------- |-----------:| ----------------- |
| **fast_hilbert** |  **0.2 ms** | Optimized for fast computation in 2D discrete space using an efficient *LUT*
| [hilbert_2d](https://crates.io/crates/hilbert_2d)      |  2.5 ms | Also allows other variants such as *Moore* and *LIU* |
| [hilbert_curve](https://crates.io/crates/hilbert_curve)      |   2.0 ms | Implements algorithm described on [Wikipedia](https://en.wikipedia.org/wiki/Hilbert_curve) |
| [hilbert](https://crates.io/crates/hilbert)      |  32.1 ms | Allows computation of higher dimensional Hilbert curves |

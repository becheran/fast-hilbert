//! Memory efficient and fast implementation of the [Hilbert space-filling curve](https://en.wikipedia.org/wiki/Hilbert_curve) computation.
//!
//! The conversion from 2D coordinates to the hilbert-curve can be described as a state diagram:
//!
//! ``` text  
//!
//! (xy)  => Discrete input coordinates in 2D space
//! [hh]  => Hilbert output for the given Input
//! # S # => State
//!
//!  ┌──────────(01) (11)◄──────────┐
//!  |┌────────►[11] [00]──────────┐|
//!  ||           # 3 #            ||  
//!  ||         (00) (10)          ||
//!  ||         [10] [01]          ||
//!  ||          |▲   ▲|           ||
//!  ▼|          └┘   └┘           ▼|
//! (01) (11)─┐           ┌─(01) (11)
//! [11] [10]◄┘           └►[01] [00]
//!   # 1 #                   # 2 #
//! (00) (10)◄┐           ┌─(00) (10)
//! [00] [01]─┘           └►[10] [11]
//!  |▲          ┌┐   ┌┐           ▲
//!  ||          |▼   ▼|          ||
//!  ||         (01) (11)         ||
//!  ||         [01] [10]         ||
//!  ||           # 0 #           ||
//!  |└─────────(00) (10)◄────────┘|
//!  └─────────►[00] [11]──────────┘
//! ```
//!
//! Instead of only processing one state-transition at a time, a pre-computed transition LUT from one state with three input values to the next
//! state is pre-computed and stored in a lookup table. The whole LUT can be packed in a 256 Byte long data-structure which fits easily in modern
//! CPU caches and allow very fast lookups without any cache misses.
//!
//! Compared to other implementations, `fast_hilbert` is at least **twice as fast** compared to other *rust* hilbert-curve implementations and uses only
//! **512 Bytes of RAM** for the lookup tables (one for 2D->1D and another for 1D->2D).
//!

#![cfg_attr(not(test), no_std)]

use core::convert::{From, TryInto};
use core::ops::{BitAnd, BitOr, BitOrAssign, Shl, ShlAssign, Shr, ShrAssign};

pub trait UnsignedBase:
    From<u8>
    + Copy
    + TryInto<usize>
    + BitOrAssign
    + BitOr<Output = Self>
    + BitAnd<Output = Self>
    + Shl<i8, Output = Self>
    + Shr<i8, Output = Self>
    + Shl<usize, Output = Self>
    + Shr<usize, Output = Self>
    + ShrAssign
    + ShlAssign
    + BitOrAssign
    + core::cmp::PartialEq
{
    fn leading_zeros(self) -> u32;
    // Save since will only be used for usize <= 8 bit for LUT lookup
    fn as_usize(self) -> usize;
    // Save since number will never exceed 8 bits
    fn as_u8(self) -> u8;
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const THREE: Self;
    const SEVEN: Self;
    const SIXTY_THREE: Self;
}

macro_rules! base_impl {
    ($T:ty) => {
        impl UnsignedBase for $T {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const TWO: Self = 2;
            const THREE: Self = 3;
            const SEVEN: Self = 7;
            const SIXTY_THREE: Self = 63;

            #[inline]
            fn leading_zeros(self) -> u32 {
                <$T>::leading_zeros(self)
            }

            #[inline]
            fn as_usize(self) -> usize {
                self as usize
            }

            #[inline]
            fn as_u8(self) -> u8 {
                self as u8
            }
        }
    };
}

base_impl!(u128);
base_impl!(u64);
base_impl!(u32);
base_impl!(u16);
base_impl!(u8);

/// Unsigned integer input type which has a double value type as key
pub trait Unsigned: UnsignedBase
where
    Self::Key: UnsignedBase,
{
    type Key; // Double the self unsigned type
}

impl Unsigned for u64 {
    type Key = u128;
}
impl Unsigned for u32 {
    type Key = u64;
}
impl Unsigned for u16 {
    type Key = u32;
}
impl Unsigned for u8 {
    type Key = u16;
}

/// Convert form 2D to 1D hilbert space.
/// Input type `T` must have half the capacity of the result type. For example (u32, u32) => u64.
///
/// # Arguments
/// * `x` - Coordinate in 2D space
/// * `y` - Coordinate in 2D space
/// * `order` - The hilbert curve order
///
/// # Examples
///```
/// let hilbert = fast_hilbert::xy2h(1u64, 0, 1);
/// assert_eq!(hilbert, 0b11u128);
///```
pub fn xy2h<T: Unsigned>(x: T, y: T, order: u8) -> <T as Unsigned>::Key {
    const LUT_3: [u8; 256] = [
        64, 1, 206, 79, 16, 211, 84, 21, 131, 2, 205, 140, 81, 82, 151, 22, 4, 199, 8, 203, 158,
        157, 88, 25, 69, 70, 73, 74, 31, 220, 155, 26, 186, 185, 182, 181, 32, 227, 100, 37, 59,
        248, 55, 244, 97, 98, 167, 38, 124, 61, 242, 115, 174, 173, 104, 41, 191, 62, 241, 176, 47,
        236, 171, 42, 0, 195, 68, 5, 250, 123, 60, 255, 65, 66, 135, 6, 249, 184, 125, 126, 142,
        141, 72, 9, 246, 119, 178, 177, 15, 204, 139, 10, 245, 180, 51, 240, 80, 17, 222, 95, 96,
        33, 238, 111, 147, 18, 221, 156, 163, 34, 237, 172, 20, 215, 24, 219, 36, 231, 40, 235, 85,
        86, 89, 90, 101, 102, 105, 106, 170, 169, 166, 165, 154, 153, 150, 149, 43, 232, 39, 228,
        27, 216, 23, 212, 108, 45, 226, 99, 92, 29, 210, 83, 175, 46, 225, 160, 159, 30, 209, 144,
        48, 243, 116, 53, 202, 75, 12, 207, 113, 114, 183, 54, 201, 136, 77, 78, 190, 189, 120, 57,
        198, 71, 130, 129, 63, 252, 187, 58, 197, 132, 3, 192, 234, 107, 44, 239, 112, 49, 254,
        127, 233, 168, 109, 110, 179, 50, 253, 188, 230, 103, 162, 161, 52, 247, 56, 251, 229, 164,
        35, 224, 117, 118, 121, 122, 218, 91, 28, 223, 138, 137, 134, 133, 217, 152, 93, 94, 11,
        200, 7, 196, 214, 87, 146, 145, 76, 13, 194, 67, 213, 148, 19, 208, 143, 14, 193, 128,
    ];
    xy2h_lut3(x, y, order, LUT_3, true, 0)
}

pub fn xy2h_moore<T: Unsigned>(x: T, y: T, order: u8) -> T::Key
where
    <T as Unsigned>::Key: From<T>,
{
    const LUT_3: [u8; 256] = [
        191, 62, 241, 176, 47, 236, 171, 42, 124, 61, 242, 115, 174, 173, 104, 41, 59, 248, 55,
        244, 97, 98, 167, 38, 186, 185, 182, 181, 32, 227, 100, 37, 69, 70, 73, 74, 31, 220, 155,
        26, 4, 199, 8, 203, 158, 157, 88, 25, 131, 2, 205, 140, 81, 82, 151, 22, 64, 1, 206, 79,
        16, 211, 84, 21, 85, 86, 89, 90, 101, 102, 105, 106, 20, 215, 24, 219, 36, 231, 40, 235,
        147, 18, 221, 156, 163, 34, 237, 172, 80, 17, 222, 95, 96, 33, 238, 111, 15, 204, 139, 10,
        245, 180, 51, 240, 142, 141, 72, 9, 246, 119, 178, 177, 65, 66, 135, 6, 249, 184, 125, 126,
        0, 195, 68, 5, 250, 123, 60, 255, 63, 252, 187, 58, 197, 132, 3, 192, 190, 189, 120, 57,
        198, 71, 130, 129, 113, 114, 183, 54, 201, 136, 77, 78, 48, 243, 116, 53, 202, 75, 12, 207,
        175, 46, 225, 160, 159, 30, 209, 144, 108, 45, 226, 99, 92, 29, 210, 83, 43, 232, 39, 228,
        27, 216, 23, 212, 170, 169, 166, 165, 154, 153, 150, 149, 213, 148, 19, 208, 143, 14, 193,
        128, 214, 87, 146, 145, 76, 13, 194, 67, 217, 152, 93, 94, 11, 200, 7, 196, 218, 91, 28,
        223, 138, 137, 134, 133, 229, 164, 35, 224, 117, 118, 121, 122, 230, 103, 162, 161, 52,
        247, 56, 251, 233, 168, 109, 110, 179, 50, 253, 188, 234, 107, 44, 239, 112, 49, 254, 127,
    ];
    if order <= 0 {
        //TODO order = 1;
    } // TOO MAX else if order > T::
    let order_minus_one = order - 1;
    let xy = (x >> order_minus_one as usize) << 1 as usize | (y >> order_minus_one as usize);
    let (h0, init_state) = if xy == T::ZERO {
        (T::Key::ZERO, 1)
    } else if xy == T::ONE {
        (T::Key::ONE, 1)
    } else if xy == T::TWO {
        (T::Key::THREE, 2)
    } else if xy == T::THREE {
        (T::Key::TWO, 2)
    } else {
        panic!("FOO")
    };
    (h0 << (order_minus_one as usize * 2))
        | xy2h_lut3(x, y, order_minus_one, LUT_3, false, init_state)
}

#[inline]
fn xy2h_lut3<T: Unsigned>(
    x: T,
    y: T,
    order: u8,
    lut3: [u8; 256],
    skip_leading_zeros: bool,
    init_state: u8,
) -> <T as Unsigned>::Key {
    let lowest_order: u8 = if skip_leading_zeros {
        let coor_bits = (core::mem::size_of::<T>() << 3) as u32;
        let useless_bits = (x | y).leading_zeros() & !1;
        (coor_bits - useless_bits) as u8 + (order & 1)
    } else {
        order
    };

    let mut result: T::Key = T::Key::ZERO;
    let mut state = init_state << 6;
    let mut shift_factor = lowest_order as i8 - 3;

    while shift_factor > 0 {
        let x_in = ((x >> shift_factor) & T::SEVEN) << 3i8;
        let y_in = (y >> shift_factor) & T::SEVEN;

        let index = (x_in | y_in | state.into()).as_usize();

        let r = lut3[index];
        state = r & 0b11000000;
        let r: T::Key = r.into();

        let mut hhh: T::Key = r & T::Key::SIXTY_THREE;
        hhh <<= ((shift_factor as u8) << 1).into();
        result |= hhh;
        shift_factor -= 3;
    }

    shift_factor *= -1;
    let x_in = ((x << shift_factor) & T::SEVEN) << 3i8;
    let y_in = (y << shift_factor) & T::SEVEN;

    let index = (x_in | y_in | state.into()).as_usize();
    let r: u8 = lut3[index];
    let r: T::Key = r.into();

    let mut hhh: T::Key = r & T::Key::SIXTY_THREE;
    hhh >>= ((shift_factor as u8) << 1).into();

    result | hhh
}

/// Convert form 1D hilbert space to 2D coordinates
///
/// Input type `T` must have double the capacity of the result types. For example u64 => (u32, u32).
///
/// # Arguments
/// * `h`     - Coordinate in 1D hilbert space
/// * `order` - Hilbert curve order
///
/// # Examples
///```
/// let (x, y) = fast_hilbert::h2xy::<u64>(0b11u128, 1);
/// assert_eq!(x, 1u64);
/// assert_eq!(y, 0u64);
///```
pub fn h2xy<T: Unsigned>(h: <T as Unsigned>::Key, order: u8) -> (T, T) {
    const LUT_3_REV: [u8; 256] = [
        64, 1, 9, 136, 16, 88, 89, 209, 18, 90, 91, 211, 139, 202, 194, 67, 4, 76, 77, 197, 70, 7,
        15, 142, 86, 23, 31, 158, 221, 149, 148, 28, 36, 108, 109, 229, 102, 39, 47, 174, 118, 55,
        63, 190, 253, 181, 180, 60, 187, 250, 242, 115, 235, 163, 162, 42, 233, 161, 160, 40, 112,
        49, 57, 184, 0, 72, 73, 193, 66, 3, 11, 138, 82, 19, 27, 154, 217, 145, 144, 24, 96, 33,
        41, 168, 48, 120, 121, 241, 50, 122, 123, 243, 171, 234, 226, 99, 100, 37, 45, 172, 52,
        124, 125, 245, 54, 126, 127, 247, 175, 238, 230, 103, 223, 151, 150, 30, 157, 220, 212, 85,
        141, 204, 196, 69, 6, 78, 79, 199, 255, 183, 182, 62, 189, 252, 244, 117, 173, 236, 228,
        101, 38, 110, 111, 231, 159, 222, 214, 87, 207, 135, 134, 14, 205, 133, 132, 12, 84, 21,
        29, 156, 155, 218, 210, 83, 203, 131, 130, 10, 201, 129, 128, 8, 80, 17, 25, 152, 32, 104,
        105, 225, 98, 35, 43, 170, 114, 51, 59, 186, 249, 177, 176, 56, 191, 254, 246, 119, 239,
        167, 166, 46, 237, 165, 164, 44, 116, 53, 61, 188, 251, 179, 178, 58, 185, 248, 240, 113,
        169, 232, 224, 97, 34, 106, 107, 227, 219, 147, 146, 26, 153, 216, 208, 81, 137, 200, 192,
        65, 2, 74, 75, 195, 68, 5, 13, 140, 20, 92, 93, 213, 22, 94, 95, 215, 143, 206, 198, 71,
    ];
    h2xy_lut3(h, order, LUT_3_REV, true, 0)
}

/// TODO
pub fn h2xy_moore<T: Unsigned>(h: <T as Unsigned>::Key, order: u8) -> (T, T) {
    const LUT_3_REV: [u8; 256] = [
        163, 226, 234, 107, 243, 187, 186, 50, 241, 185, 184, 48, 168, 169, 97, 96, 167, 230, 238,
        111, 247, 191, 190, 54, 245, 189, 188, 52, 172, 173, 101, 100, 156, 157, 85, 84, 12, 68,
        69, 205, 14, 70, 71, 207, 151, 214, 222, 95, 152, 153, 81, 80, 8, 64, 65, 201, 10, 66, 67,
        203, 147, 210, 218, 91, 241, 185, 184, 48, 243, 187, 186, 50, 42, 98, 99, 235, 40, 96, 97,
        233, 152, 153, 81, 80, 8, 64, 65, 201, 10, 66, 67, 203, 147, 210, 218, 91, 156, 157, 85,
        84, 12, 68, 69, 205, 14, 70, 71, 207, 151, 214, 222, 95, 231, 175, 174, 38, 165, 228, 236,
        109, 181, 244, 252, 125, 62, 118, 119, 255, 199, 143, 142, 6, 133, 196, 204, 77, 149, 212,
        220, 93, 30, 86, 87, 223, 167, 230, 238, 111, 247, 191, 190, 54, 245, 189, 188, 52, 172,
        173, 101, 100, 163, 226, 234, 107, 243, 187, 186, 50, 241, 185, 184, 48, 168, 169, 97, 96,
        209, 153, 152, 16, 211, 155, 154, 18, 10, 66, 67, 203, 8, 64, 65, 201, 135, 198, 206, 79,
        215, 159, 158, 22, 213, 157, 156, 20, 140, 141, 69, 68, 195, 139, 138, 2, 129, 192, 200,
        73, 145, 208, 216, 89, 26, 82, 83, 219, 227, 171, 170, 34, 161, 224, 232, 105, 177, 240,
        248, 121, 58, 114, 115, 251, 188, 189, 117, 116, 44, 100, 101, 237, 46, 102, 103, 239, 183,
        246, 254, 127,
    ];
    h2xy_lut3(h, order, LUT_3_REV, false, 1 / 2)
}

#[inline]
fn h2xy_lut3<T: Unsigned>(
    h: <T as Unsigned>::Key,
    order: u8,
    lut3: [u8; 256],
    skip_leading_zeros: bool,
    init_state: u8,
) -> (T, T) {
    let lowest_order: u8 = if skip_leading_zeros {
        let coor_bits = (core::mem::size_of::<T>() << 3) as u8;
        let useless_bits = (h.leading_zeros() >> 1) as u8 & !1;
        coor_bits - useless_bits + (order & 1)
    } else {
        order // TODO first iteration is default route
    };

    let mut x_result: T = T::ZERO;
    let mut y_result: T = T::ZERO;

    let mut state = init_state;
    let mut shift_factor = lowest_order as i8 - 3;

    while shift_factor > 0 {
        let h_in: T::Key = h >> ((shift_factor as usize) << 1);
        let h_in: T::Key = h_in & T::Key::SIXTY_THREE;
        let h_in: u8 = h_in.as_u8();

        let r: u8 = lut3[state as usize | h_in as usize];
        state = r & 0b11000000;

        let xxx: T = r.into();
        let xxx: T = xxx >> 3i8;
        let xxx: T = xxx & T::SEVEN;

        let yyy: T = r.into();
        let yyy: T = yyy & T::SEVEN;

        x_result |= xxx << shift_factor;
        y_result |= yyy << shift_factor;
        shift_factor -= 3;
    }

    shift_factor *= -1;
    let h_in: T::Key = h << ((shift_factor as usize) << 1);
    let h_in: T::Key = h_in & T::Key::SIXTY_THREE;
    let h_in: u8 = h_in.as_u8();

    let r: u8 = lut3[state as usize | h_in as usize];

    let xxx: T = r.into();
    let xxx: T = xxx >> 3i8;
    let xxx: T = xxx & T::SEVEN;

    let yyy: T = r.into();
    let yyy: T = yyy & T::SEVEN;

    x_result = xxx >> shift_factor | x_result;
    y_result = yyy >> shift_factor | y_result;

    (x_result, y_result)
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate image;

    #[test]
    fn hilbert_and_rev_full_order() {
        let order = 8;
        let max = 2usize.pow(order * 2);
        for h in 0..max {
            let (x, y): (u8, u8) = h2xy(h as u16, order as u8);
            let res_h = xy2h(x, y, order as u8);
            assert_eq!(h as u16, res_h);
        }
    }

    #[test]
    fn xy2h_hilbert() {
        for &order in &[1, 2, 3, 4, 5, 8, 13, 16] {
            for x in 0..order * 2 {
                for y in 0..order * 2 {
                    let d = hilbert_2d::xy2h_discrete(x, y, order, hilbert_2d::Variant::Hilbert);
                    let df = xy2h(x as u32, y as u32, order as u8);
                    assert_eq!(d as u64, df);
                }
            }
        }
    }

    #[test]
    fn xy2h_moore_test() {
        for &order in &[1, 2, 3, 4, 5, 8, 13, 16] {
            for x in 0..order * 2 {
                for y in 0..order * 2 {
                    let d = hilbert_2d::xy2h_discrete(x, y, order, hilbert_2d::Variant::Moore);
                    let df = xy2h_moore(x as u32, y as u32, order as u8);
                    assert_eq!(d as u64, df);
                }
            }
        }
    }

    #[test]
    fn h2xy_hilbert() {
        for &order in &[1, 2, 3, 5, 8] {
            let numbers = 4usize.pow(order);
            for d in 0..numbers {
                let (x, y) =
                    hilbert_2d::h2xy_discrete(d, order as usize, hilbert_2d::Variant::Hilbert);
                let (x_fast, y_fast): (u32, u32) = h2xy(d as u64, order as u8);
                assert_eq!((x, y), (x_fast as usize, y_fast as usize));
            }
        }
    }

    #[test]
    fn h2xy_moore_test() {
        for &order in &[1, 2, 3, 5, 8] {
            let numbers = 4usize.pow(order);
            for d in 0..numbers {
                let (x, y) =
                    hilbert_2d::h2xy_discrete(d, order as usize, hilbert_2d::Variant::Moore);
                let (x_fast, y_fast): (u32, u32) = h2xy_moore(d as u64, order as u8);
                assert_eq!((x, y), (x_fast as usize, y_fast as usize));
            }
        }
    }

    fn draw_hilbert_curve(iteration: u32) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
        let size: usize = 256;
        let border = 32 / iteration;

        let mut imgbuf = image::ImageBuffer::new(size as u32, size as u32);

        let mut points: Vec<(u32, u32)> = vec![(0, 0); 2usize.pow(iteration * 2)];
        for i in 0..2usize.pow(iteration * 2) {
            let (mut x, mut y) = h2xy(i as u64, iteration as u8);
            let step = (size as u32 - border * 2) as f64 / (2usize.pow(iteration) as f64 - 1.0);
            x = (x as f64 * step) as u32 + border;
            y = (y as f64 * step) as u32 + border;
            points[i] = (x, y)
        }

        let mut prev = (0, 0);
        let white = image::Rgb([255_u8, 255, 255]);

        for (x, y) in &points {
            if prev == (0, 0) {
                prev = (*x, *y);
                continue;
            }
            while prev.0 < *x {
                let pixel = imgbuf.get_pixel_mut(prev.0 as u32, prev.1 as u32);
                *pixel = white;
                prev.0 += 1;
                continue;
            }
            while prev.0 > *x {
                let pixel = imgbuf.get_pixel_mut(prev.0 as u32, prev.1 as u32);
                *pixel = white;
                prev.0 -= 1;
                continue;
            }
            while prev.1 < *y {
                let pixel = imgbuf.get_pixel_mut(prev.0 as u32, prev.1 as u32);
                *pixel = white;
                prev.1 += 1;
                continue;
            }
            while prev.1 > *y {
                let pixel = imgbuf.get_pixel_mut(prev.0 as u32, prev.1 as u32);
                *pixel = white;
                prev.1 -= 1;
                continue;
            }
        }
        imgbuf
    }

    // Only for rendering images
    #[test]
    fn write_image() {
        for i in 1..7 {
            let imgbuf = draw_hilbert_curve(i);
            imgbuf.save(format!("doc/h{}.png", i)).unwrap();
        }
    }

    #[test]
    fn gen_lut3() {
        // From 2D to 1D
        // 4 bits => 4 bits
        const LUT_SXY2SH_HILBERT: [u8; 16] = [4, 1, 11, 2, 0, 15, 5, 6, 10, 9, 3, 12, 14, 7, 13, 8];
        const LUT_SH2SXY_HILBERT: [u8; 16] = [
            //  0         1        2       3    <- H
            0b01_00, 0b00_01, 0b00_11, 0b10_10, // S0
            0b00_00, 0b01_10, 0b01_11, 0b11_01, // S1
            0b11_11, 0b10_01, 0b10_00, 0b00_10, // S2
            0b10_11, 0b11_10, 0b11_00, 0b01_01, // S3
        ];
        const LUT_SXY2SH_MOORE: [u8; 16] = [
            //  00       01       10      11    <- XY
            0b10_11, 0b00_10, 0b01_00, 0b00_01, // S0
            0b01_01, 0b01_10, 0b00_00, 0b11_11, // S1
            0b00_11, 0b11_00, 0b10_10, 0b10_01, // S2
            0b11_01, 0b10_00, 0b11_10, 0b01_11, // S3
        ];
        const LUT_SH2SXY_MOORE: [u8; 16] = [
            //  0         1        2       3    <- H
            0b10_10, 0b10_11, 0b01_01, 0b01_00, // S0
            0b00_10, 0b01_00, 0b01_01, 0b11_11, // S1
            0b11_01, 0b10_11, 0b10_10, 0b00_00, // S2
            0b10_01, 0b11_00, 0b11_10, 0b01_11, // S3
        ];
        println!("Hilbert sxxxyyy2shhh");
        gen_lut3_sxxxyyy2shhh(LUT_SXY2SH_HILBERT);
        println!("Hilbert shhh2sxxxyyy");
        gen_lut3_shhh2sxxxyyy(LUT_SH2SXY_HILBERT);
        println!("Moore sxxxyyy2shhh");
        gen_lut3_sxxxyyy2shhh(LUT_SXY2SH_MOORE);
        println!("Moore shhh2sxxxyyy");
        gen_lut3_shhh2sxxxyyy(LUT_SH2SXY_MOORE)
    }

    fn gen_lut3_sxxxyyy2shhh(lut_sxy2sh: [u8; 16]) {
        // State 0, 1, 2, 3
        let mut lut_3: [u8; 256] = [0; 256];
        for input in 0..=255 {
            //for input in 4..=4 {
            let mut state: u8 = (input as u8 & 0b11000000) >> 4;
            let mut result: u8 = 0;
            let mut x_mask: u8 = 0b00100000;
            let mut y_mask: u8 = 0b00000100;
            for i in 0..3 {
                let idx = state | (input & x_mask) >> (4 - i) | (input & y_mask) >> (2 - i);
                let r = lut_sxy2sh[idx as usize];
                // Override State
                state = r & 0b1100;
                result = (result & 0b00111111) | (state << 4);
                // Dx Dy
                result = (result & !(0b00110000 >> (i * 2))) | ((r & 0b0011) << ((2 - i) * 2));
                x_mask >>= 1;
                y_mask >>= 1;
            }
            lut_3[input as usize] = result;
        }
        println!("{:?}", lut_3);
    }

    fn gen_lut3_shhh2sxxxyyy(lut_sh2sxy: [u8; 16]) {
        // State 0, 1, 2, 3
        let mut lut_3: [u8; 256] = [0; 256];
        for input in 0..=255 {
            //for input in 4..=4 {
            let mut state: u8 = (input as u8 & 0b11000000) >> 6;
            let mut result: u8 = 0;
            let mut h_mask: u8 = 0b00110000;
            for i in 0..3 {
                let idx = (state << 2) | (input & h_mask) >> (4 - (i * 2));
                let r = lut_sh2sxy[idx as usize];
                // Override State
                state = (r & 0b1100) >> 2;
                let x = (r & 0b10) >> 1;
                let y = r & 0b1;
                // Set state
                result = (result & 0b00111111) | (state << 6);
                result = (result & !(0b00100000 >> i)) | (x << (5 - i));
                result = (result & !(0b00000100 >> i)) | (y << (2 - i));
                h_mask >>= 2;
            }
            lut_3[input as usize] = result;
        }
        println!("{:?}", lut_3);
    }
}

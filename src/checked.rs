use crate::{h2xy, xy2h, Unsigned, UnsignedBase};

/// The main error type for this crate
#[derive(Debug, PartialEq, Eq)]
pub enum OrderError<T: Unsigned> {
    /// The given order did not fit into the type
    InvalidOrder {
        order: u8,
        the_type: &'static str,
        coor_bits: u8,
    },
    /// The given coordinates did not fit into the given order
    OrderExceeded {
        order: u8,
        max_allowed_index: T::Key,
        given_index: T::Key,
    },
}

impl<T: Unsigned> core::fmt::Display for OrderError<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OrderError::InvalidOrder {
                order,
                the_type,
                coor_bits,
            } => write!(
                f,
                "Type {} can at most support order {}, which {} exceeds",
                the_type,
                coor_bits,
                order,
            ),
            OrderError::OrderExceeded { order, max_allowed_index, given_index } => write!(f, "order {order} can at most index to {max_allowed_index:?}, which {given_index:?} exceeds"),
        }
    }
}
impl<T: Unsigned> core::error::Error for OrderError<T> {}

/// The maximum allowed `x` or `y` coordinate for a given hilbert order
///
/// returns [`OrderError`] if the order doesn't fit in the current type
///
/// ```
/// use fast_hilbert::max_coord;
/// assert_eq!(max_coord::<u8>(1).unwrap(),1);
/// assert_eq!(max_coord::<u8>(8).unwrap(),255);
/// assert!(max_coord::<u8>(9).is_err())
/// ```
pub fn max_coord<T: Unsigned>(order: u8) -> Result<T::Key, OrderError<T>> {
    let coor_bits = (size_of::<T>() << 3) as u8;
    // possible coordinates/index:
    // h in [0,1<<2*order)
    // (x,y) in [0,1<<order)
    // therefore, order must <= size_of::<T>
    if order > coor_bits {
        Err(OrderError::InvalidOrder {
            order,
            the_type: core::any::type_name::<T>(),
            coor_bits,
        })
    } else {
        // if we have the maximum number of bits in the key,
        // e.g. order 8 where T = u8,
        // then the maximum value for h is 4.pow(8) = (1<<16)-1
        // but 1<<coor_bits breaks for T, so we use T::Key
        Ok((T::Key::from(1) << usize::from(order)) - 1.into())
    }
}

/// The maximum allowed hilbert index for a given order
///
/// returns [`OrderError`] if the order doesn't fit in the current type
///
/// ```
/// use fast_hilbert::max_index;
/// assert_eq!(max_index::<u8>(1).unwrap(),3);
/// assert_eq!(max_index::<u8>(8).unwrap(),u16::MAX);
/// assert!(max_index::<u8>(9).is_err())
/// ```
pub fn max_index<T: Unsigned>(order: u8) -> Result<T::Key, OrderError<T>> {
    let coor_bits = (size_of::<T>() << 3) as u8;
    // possible coordinates/index:
    // h in [0,1<<2*order)
    // (x,y) in [0,1<<order)
    // therefore, order must <= size_of::<T>
    if order > coor_bits {
        return Err(OrderError::InvalidOrder {
            order,
            the_type: core::any::type_name::<T>(),
            coor_bits,
        });
    } else {
        // if we have the maximum number of bits in the key,
        // e.g. order 8 where T = u8,
        // then the maximum value for h is 4.pow(8) = (1<<16)-1
        // but that breaks maths on T::Key = u16
        // so we special-case the maximum
        Ok(if order == coor_bits {
            !T::Key::ZERO
        } else {
            (T::Key::from(1) << usize::from(order * 2)) - 1.into()
        })
    }
}

/// The checked version of [`xy2h`].
///
/// returns [`OrderError`] if the order doesn't fit in the current type, or the `(x,y)` coordinates don't fit in the order.
///
/// ```
/// use fast_hilbert::xy2h_checked;
/// assert_eq!(xy2h_checked::<u8>(1,1,1).unwrap(),2);
/// // the coordinate doesn't fit in the order
/// assert!(xy2h_checked::<u8>(1,2,1).is_err());
/// assert!(xy2h_checked::<u8>(2,1,1).is_err());
/// // the order doesn't fit in the type
/// assert!(xy2h_checked::<u8>(2,1,9).is_err());
/// ```
pub fn xy2h_checked<T: Unsigned>(
    x: T,
    y: T,
    order: u8,
) -> Result<<T as Unsigned>::Key, OrderError<T>> {
    let max_coord = max_coord(order)?;
    if (x | y).into() > max_coord {
        Err(OrderError::OrderExceeded {
            order,
            max_allowed_index: max_coord,
            given_index: x.max(y).into(),
        })
    } else {
        Ok(xy2h(x, y, order))
    }
}

/// The checked version of [`h2xy`].
///
/// returns [`OrderError`] if the order doesn't fit in the current type, or the hilbert index doesn't fit in the order.
///
/// ```
/// use fast_hilbert::h2xy_checked;
/// assert_eq!(h2xy_checked::<u8>(3,1).unwrap(),(1,0));
/// // the index doesn't fit in the order
/// assert!(h2xy_checked::<u8>(4,1).is_err());
/// // the order doesn't fit in the type
/// assert!(h2xy_checked::<u8>(4,9).is_err());
/// ```
pub fn h2xy_checked<T: Unsigned>(
    h: <T as Unsigned>::Key,
    order: u8,
) -> Result<(T, T), OrderError<T>> {
    let max_index = max_index(order)?;
    if h > max_index {
        return Err(OrderError::OrderExceeded {
            order,
            max_allowed_index: max_index,
            given_index: h,
        });
    }
    Ok(h2xy(h, order))
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        assert_eq!(
            &OrderError::InvalidOrder::<u8> {
                order: 9,
                the_type: "u8",
                coor_bits: 8
            }
            .to_string(),
            "Type u8 can at most support order 8, which 9 exceeds"
        );
        assert_eq!(
            &OrderError::OrderExceeded::<u8> {
                order: 1,
                max_allowed_index: 3,
                given_index: 4
            }
            .to_string(),
            "order 1 can at most index to 3, which 4 exceeds"
        );
    }

    #[test]
    fn test_invalid_order_xy2h_checked() {
        // Test that invalid order returns the appropriate error
        // For u8 coordinates, order should not exceed 16 (8 bits * 2)
        assert_eq!(xy2h_checked(5u8, 10u8, 8), Ok(119));
        assert_eq!(
            xy2h_checked(5u8, 10u8, 9).unwrap_err(),
            OrderError::InvalidOrder {
                order: 9,
                the_type: &"u8",
                coor_bits: 8
            }
        );
        assert_eq!(
            xy2h_checked(5u8, 10u8, 255).unwrap_err(),
            OrderError::InvalidOrder {
                order: 255,
                the_type: &"u8",
                coor_bits: 8
            }
        );

        // For u32 coordinates, order should not exceed 64 (32 bits * 2)
        assert!(xy2h_checked(100u32, 200u32, 32).is_ok());
        assert_eq!(
            xy2h_checked(100u32, 200u32, 33).unwrap_err(),
            OrderError::InvalidOrder {
                order: 33,
                the_type: &"u32",
                coor_bits: 32
            }
        );
    }

    #[test]
    fn test_invalid_order_h2xy_checked() {
        // Test that invalid order returns the appropriate error
        // For u8 coordinates, order should not exceed 16 (8 bits * 2)
        assert_eq!(h2xy_checked::<u8>(100u16, 8).unwrap(), (4, 14));
        assert_eq!(
            h2xy_checked::<u8>(100u16, 9).unwrap_err(),
            OrderError::InvalidOrder {
                order: 9,
                the_type: &"u8",
                coor_bits: 8
            }
        );
        assert_eq!(
            h2xy_checked::<u8>(100u16, 255).unwrap_err(),
            OrderError::InvalidOrder {
                order: 255,
                the_type: &"u8",
                coor_bits: 8
            }
        );

        // For u32 coordinates, order should not exceed 64 (32 bits * 2)
        assert_eq!(h2xy_checked::<u32>(1000u64, 32).unwrap(), (6, 30));
        assert_eq!(
            h2xy_checked::<u32>(1000u64, 33).unwrap_err(),
            OrderError::InvalidOrder {
                order: 33,
                the_type: &"u32",
                coor_bits: 32
            }
        );
    }

    #[test]
    fn test_exceeds_order() {
        // a 1-order hilbert curve has 4 points, spanning [0,1]x[0,1]-space
        assert_eq!(h2xy_checked::<u64>(3, 1).unwrap(), (1, 0));
        // therefore, when 0-indexed, 4 should return an error.
        // in stead, it returns (2,0) which is not inside the space.
        // I think it secretly upgraded the hilbert order
        assert_eq!(
            h2xy_checked::<u64>(4, 1),
            Err(OrderError::OrderExceeded {
                order: 1,
                max_allowed_index: 3,
                given_index: 4
            })
        );

        assert_eq!(xy2h_checked::<u64>(1, 1, 1).unwrap(), 2);
        assert_eq!(
            xy2h_checked::<u64>(1, 2, 1).unwrap_err(),
            OrderError::OrderExceeded {
                order: 1,
                max_allowed_index: 1,
                given_index: 2
            }
        );
        assert_eq!(
            xy2h_checked::<u64>(2, 1, 1).unwrap_err(),
            OrderError::OrderExceeded {
                order: 1,
                max_allowed_index: 1,
                given_index: 2
            }
        );
    }

    #[test]
    fn test_edge_cases() {
        // Test with maximum values
        assert_eq!(
            xy2h_checked(u64::MAX, u64::MAX, 64),
            Ok(226854911280625642308916404954512140970)
        );
        assert_eq!(
            h2xy_checked::<u64>(u128::MAX, 64).unwrap(),
            (18446744073709551615, 0)
        );

        // Test with zero values
        assert_eq!(xy2h_checked(0u32, 0u32, 0).unwrap(), 0);
        assert_eq!(h2xy_checked::<u32>(0u64, 0).unwrap(), (0, 0));

        // Test that we handle untrusted data gracefully
        assert!(xy2h_checked(u32::MAX, u32::MAX, u8::MAX).is_err());
        assert!(h2xy_checked::<u32>(u64::MAX, u8::MAX).is_err());
    }
}

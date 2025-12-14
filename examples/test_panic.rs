use fast_hilbert::{xy2h, h2xy};

fn main() {
    // Test with max values
    println!("Testing with max u64 values...");
    let result = xy2h(u64::MAX, u64::MAX, 64);
    println!("xy2h(u64::MAX, u64::MAX, 64) = {}", result);
    
    // Test with 0 order
    println!("\nTesting with order 0...");
    let result2 = xy2h(5u32, 10u32, 0);
    println!("xy2h(5, 10, 0) = {}", result2);
    
    // Test h2xy with max value
    println!("\nTesting h2xy with max u128...");
    let (x, y) = h2xy::<u64>(u128::MAX, 64);
    println!("h2xy(u128::MAX, 64) = ({}, {})", x, y);
    
    // Test with large order
    println!("\nTesting with order > bit capacity...");
    let result3 = xy2h(5u8, 10u8, 255);
    println!("xy2h(5u8, 10u8, 255) = {}", result3);
    
    println!("\nAll tests completed!");
}

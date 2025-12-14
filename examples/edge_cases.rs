use fast_hilbert::{xy2h, h2xy};

fn main() {
    // Edge case: coordinates that exceed the order
    // For order 2, max coordinate should be 3 (2^2 - 1)
    // But what if we provide larger coordinates?
    
    println!("Test: coordinates larger than order allows");
    println!("Order 2 means 2x2 bits, so coords should be 0-3");
    println!("But providing x=100, y=200:");
    let result = xy2h(100u32, 200u32, 2);
    println!("Result: {}", result);
    
    println!("\nTest: order 0");
    let result = xy2h(0u32, 0u32, 0);
    println!("xy2h(0, 0, 0) = {}", result);
    
    println!("\nTest: large h value for small order");
    let (x, y) = h2xy::<u32>(1000000u64, 4);
    println!("h2xy(1000000, 4) = ({}, {})", x, y);
}

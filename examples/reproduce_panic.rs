use fast_hilbert::{xy2h, h2xy};

fn main() {
    // Try to trigger a panic with invalid order values
    // The order should be reasonable (e.g., 1-64 for u64), 
    // but user-supplied data might be invalid
    
    println!("Test 1: Very large order value (should work based on earlier test)");
    let result = xy2h(5u8, 10u8, 200);
    println!("xy2h(5u8, 10u8, 200) = {}", result);
    
    println!("\nTest 2: Large coordinates");
    let result = xy2h(200u8, 250u8, 10);
    println!("xy2h(200u8, 250u8, 10) = {}", result);
}

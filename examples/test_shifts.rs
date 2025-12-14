fn main() {
    // Simulate what happens in xy2h
    let x: u8 = 255;
    let y: u8 = 255;
    let order: u8 = 100;
    
    let coor_bits = (core::mem::size_of::<u8>() << 3) as u32; // 8
    let useless_bits = (x | y).leading_zeros() & !1;  // 0 & !1 = 0
    let lowest_order = (coor_bits - useless_bits) as u8 + (order & 1); // 8 + 0 = 8
    
    println!("coor_bits: {}", coor_bits);
    println!("useless_bits: {}", useless_bits);
    println!("lowest_order: {}", lowest_order);
    
    let mut shift_factor = lowest_order as i8 - 3; // 8 - 3 = 5
    println!("Initial shift_factor: {}", shift_factor);
    
    // First loop iteration
    if shift_factor > 0 {
        println!("\nFirst iteration:");
        let x_in = ((x >> shift_factor) & 7) << 3i8;
        println!("x_in: {}", x_in);
        
        shift_factor -= 3;
        println!("shift_factor after first iteration: {}", shift_factor);
    }
    
    // Second loop iteration  
    if shift_factor > 0 {
        println!("\nSecond iteration:");
        let x_in = ((x >> shift_factor) & 7) << 3i8;
        println!("x_in: {}", x_in);
        
        shift_factor -= 3;
        println!("shift_factor after second iteration: {}", shift_factor);
    }
    
    // After the loop - the problematic part
    println!("\nAfter loop:");
    shift_factor *= -1;
    println!("shift_factor negated: {}", shift_factor);
    
    // This is line 186 in lib.rs
    println!("\nThis is the potentially problematic line:");
    println!("Attempting: (({} << {}) & 7) << 3", x, shift_factor);
    
    // Let's check if this could overflow
    // For u8, max safe left shift is 7 (255 << 7 = ...)
    if shift_factor > 7 {
        println!("WARNING: shift_factor {} is too large for u8!", shift_factor);
    }
    
    let x_in = ((x << shift_factor) & 7) << 3i8;
    println!("x_in: {}", x_in);
}

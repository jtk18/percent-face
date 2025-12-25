use std::env;
use std::fs::File;
use std::io::{BufReader, Read};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.dat>", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];
    println!("Loading model: {}", path);

    let file = File::open(path).expect("Failed to open file");
    let mut reader = BufReader::new(file);

    // Read first 256 bytes and analyze
    let mut header = [0u8; 256];
    reader.read_exact(&mut header).expect("Failed to read header");

    println!("\nFirst 256 bytes:");
    for (i, chunk) in header.chunks(16).enumerate() {
        print!("{:04x}: ", i * 16);
        for b in chunk {
            print!("{:02x} ", b);
        }
        print!(" ");
        for b in chunk {
            if *b >= 32 && *b < 127 {
                print!("{}", *b as char);
            } else {
                print!(".");
            }
        }
        println!();
    }

    // Try to decode dlib's varint format
    println!("\nAttempting varint decode:");

    let mut pos = 0;

    fn read_varint(data: &[u8], pos: &mut usize) -> Option<u64> {
        // dlib uses a custom variable-length encoding:
        // If high bit is 0, it's a single byte
        // If high bit is 1, the lower 7 bits indicate how many more bytes follow
        if *pos >= data.len() {
            return None;
        }

        let first = data[*pos];
        *pos += 1;

        if first & 0x80 == 0 {
            // Single byte value
            Some(first as u64)
        } else {
            // Multi-byte: lower 7 bits = number of following bytes
            let num_bytes = (first & 0x7F) as usize;
            if *pos + num_bytes > data.len() {
                return None;
            }

            let mut value: u64 = 0;
            for i in 0..num_bytes {
                value |= (data[*pos + i] as u64) << (i * 8);
            }
            *pos += num_bytes;
            Some(value)
        }
    }

    // Decode first several varints
    for i in 0..20 {
        if pos >= header.len() {
            break;
        }
        let start = pos;
        if let Some(val) = read_varint(&header, &mut pos) {
            println!("  varint[{}] at {:#04x}: {} ({:#x})", i, start, val, val);
        } else {
            println!("  varint[{}] at {:#04x}: FAILED", i, start);
            break;
        }
    }

    // Also try loading with our dlib loader
    println!("\nTrying percent_face::dlib::load_dlib_model...");
    match percent_face::dlib::load_dlib_model(path) {
        Ok(model) => {
            println!("SUCCESS! Model loaded:");
            println!("  num_landmarks: {}", model.num_landmarks());
            println!("  cascade_stages: {}", model.num_cascade_stages());
        }
        Err(e) => {
            println!("FAILED: {}", e);
        }
    }
}

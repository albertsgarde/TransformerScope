use std::env;

use transformer_scope::{html, Payload};

pub fn main() {
    let payload_path = env::args().nth(1).unwrap();
    let site_path = env::args().nth(2).unwrap();

    let payload = Payload::from_dir(payload_path);

    let start_time = std::time::Instant::now();
    html::generate_site_in_dir(site_path, &payload);
    println!("Site generated in {:?}", start_time.elapsed());
}

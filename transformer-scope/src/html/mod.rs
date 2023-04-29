use std::{fs, path::Path};

use fs_extra::dir::CopyOptions;

mod index;
pub mod template;
pub use index::generate_index_page;
mod neuron;
pub use neuron::generate_neuron_page;
mod board_heatmap;
pub use board_heatmap::board_heatmap;

use crate::Payload;

fn generate_site_in_dir_inner(path: &Path, payload: &Payload) {
    fs::create_dir_all(path).unwrap();

    let index_page = generate_index_page(payload);
    fs::write(path.join("index.html"), index_page.into_string()).unwrap();

    // Copy static files.
    let static_path = path.join("static");
    fs_extra::copy_items(
        &["static"],
        &static_path,
        &CopyOptions::new().overwrite(true).copy_inside(true),
    )
    .unwrap();

    // Generate site.
    let ownership_heatmaps = payload.ownership_heatmaps();
    let (layer_count, neuron_count, _, _) = ownership_heatmaps.dim();
    for layer_index in 0..layer_count {
        println!("Generating pages for neurons in layer {layer_index}...");
        let layer_path = path.join(format!("L{layer_index}"));
        fs::create_dir(&layer_path).unwrap();
        for neuron_index in 0..neuron_count {
            let neuron_page = generate_neuron_page(layer_index, neuron_index, payload);
            let neuron_path = layer_path.join(format!("N{neuron_index}.html"));
            fs::write(neuron_path, neuron_page.into_string()).unwrap();
        }
    }
}

pub fn generate_site_in_dir<P: AsRef<Path>>(path: P, payload: &Payload) {
    generate_site_in_dir_inner(path.as_ref(), payload);
}

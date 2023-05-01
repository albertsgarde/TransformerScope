use std::{
    fs::{self},
    path::Path,
};

use crate::{html, Payload};

const STYLE_CSS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../static/style.css"));

fn generate_site_in_dir_inner(path: &Path, payload: &Payload) {
    fs::create_dir_all(path).unwrap();

    let index_page = html::index::generate_index_page(payload, true);
    fs::write(path.join("index.html"), index_page.into_string()).unwrap();

    // Copy static files.
    let static_path = path.join("static");
    fs::create_dir_all(&static_path).unwrap();
    fs::write(static_path.join("style.css"), STYLE_CSS).unwrap();

    // Generate site.
    let layer_count = payload.num_layers();
    let neuron_count = payload.num_neurons();
    for layer_index in 0..layer_count {
        println!("Generating pages for neurons in layer {layer_index}...");
        let layer_path = path.join(format!("L{layer_index}"));
        fs::create_dir(&layer_path).unwrap();
        for neuron_index in 0..neuron_count {
            let neuron_page =
                html::neuron::generate_neuron_page(layer_index, neuron_index, payload, true);
            let neuron_path = layer_path.join(format!("N{neuron_index}.html"));
            fs::write(neuron_path, neuron_page.into_string()).unwrap();
        }
    }
}

pub fn generate_site_in_dir<P: AsRef<Path>>(path: P, payload: &Payload) {
    generate_site_in_dir_inner(path.as_ref(), payload);
}

use maud::Markup;

use crate::Payload;

pub fn generate_neuron_page(
    layer_index: usize,
    neuron_index: usize,
    payload: &Payload,
    file: bool,
) -> Markup {
    payload
        .neuron_template()
        .generate(payload, file, layer_index, neuron_index)
}

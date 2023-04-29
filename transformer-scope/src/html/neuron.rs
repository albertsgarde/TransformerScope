use maud::Markup;

use crate::Payload;

pub fn generate_neuron_page(layer_index: usize, neuron_index: usize, payload: &Payload) -> Markup {
    payload
        .neuron_template()
        .generate(payload, layer_index, neuron_index)
}

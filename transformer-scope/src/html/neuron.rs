use maud::{html, Markup};
use ndarray::s;

use crate::Payload;

pub fn generate_neuron_page(layer_index: usize, neuron_index: usize, payload: &Payload) -> Markup {
    let ownership_heatmaps = payload.ownership_heatmaps();

    let values = ownership_heatmaps
        .slice(s![layer_index, neuron_index, .., ..])
        .to_owned();

    let neuron_rank = payload
        .neuron_ranks()
        .get((layer_index, neuron_index))
        .unwrap();

    let heatmap_html = super::board_heatmap(&values);
    html! {
        head {
            meta charset="utf-8";
            title { "Transformer Scope - Layer " (layer_index) " Neuron " (neuron_index)}
            link rel="stylesheet" href="/static/style.css"{};
        }
        (heatmap_html)
        p {"This neuron has rank " (neuron_rank) " variance within the layer."}
    }
}

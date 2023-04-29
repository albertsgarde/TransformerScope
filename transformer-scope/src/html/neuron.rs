use maud::{html, Markup};
use ndarray::s;

use crate::ApplicationState;

pub fn neuron_html(layer_index: usize, neuron_index: usize, state: &ApplicationState) -> Markup {
    let ownership_heatmaps = state.ownership_heatmaps();

    let values = ownership_heatmaps
        .slice(s![layer_index, neuron_index, .., ..])
        .to_owned();

    let heatmap_html = super::board_heatmap(&values);
    html! {
        head {
            meta charset="utf-8";
            title { "Transformer Scope - Layer " (layer_index) " Neuron " (neuron_index)}
            link rel="stylesheet" href="/static/style.css"{};
        }
        (heatmap_html)
    }
}

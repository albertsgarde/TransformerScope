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

    let previous_neuron_link = if neuron_index > 0 {
        html! {
            a href={"N"({neuron_index-1})} {
                "Previous"
            }
        }
    } else if layer_index > 0 {
        html! {
            a href={"../L"({layer_index-1})"/N"({ownership_heatmaps.dim().1-1})} {
                "Previous layer"
            }
        }
    } else {
        html! {}
    };

    let next_neuron_link = if neuron_index < ownership_heatmaps.dim().1 - 1 {
        html! {
            a href={"N"({neuron_index+1})} {
                "Next"
            }
        }
    } else if layer_index < ownership_heatmaps.dim().0 - 1 {
        html! {
            a href={"../L"({layer_index+1})"/N0"} {
                "Next layer"
            }
        }
    } else {
        html! {}
    };

    let heatmap_html = super::board_heatmap(&values);
    html! {
        head {
            meta charset="utf-8";
            title { "Transformer Scope - Layer " (layer_index) " Neuron " (neuron_index)}
            link rel="stylesheet" href="/static/style.css"{};
        }
        a href="/" {"Back to index"}
        h1 {"Transformer Scope - Layer " (layer_index) " Neuron " (neuron_index)}
        (previous_neuron_link)" - "(next_neuron_link)
        div class="columns" {
            div class="column" {
                h3 {"Ownership heatmap"}
                (heatmap_html)
            }
        }
        p {"This neuron has rank " (neuron_rank) " variance within the layer."}
    }
}

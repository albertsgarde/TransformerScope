use maud::{html, Markup};
use ndarray::s;

use crate::Payload;

pub fn generate_neuron_page(layer_index: usize, neuron_index: usize, payload: &Payload) -> Markup {
    let ownership_heatmaps = payload.ownership_heatmaps();
    let blank_heatmaps = payload.blank_heatmaps();

    let ownership_heatmap_html =
        super::board_heatmap(ownership_heatmaps.slice(s![layer_index, neuron_index, .., ..]));
    let blank_heatmap_html =
        super::board_heatmap(blank_heatmaps.slice(s![layer_index, neuron_index, .., ..]));

    let neuron_rank = payload
        .neuron_ranks()
        .get((layer_index, neuron_index))
        .unwrap();

    html! {
        head {
            meta charset="utf-8";
            title { "Transformer Scope - Layer " (layer_index) " Neuron " (neuron_index)}
            link rel="stylesheet" href="/static/style.css"{};
        }
        a href="/" {"Back to index"}
        h1 {"Transformer Scope - Layer " (layer_index) " Neuron " (neuron_index)}
        (generate_navigation_links(ownership_heatmaps.dim().0, ownership_heatmaps.dim().1, layer_index, neuron_index))
        div class="columns" {
            div class="column" {
                h3 {"Ownership heatmap"}
                (ownership_heatmap_html)
            }
            div class="column" {
                h3 {"Blank heatmap"}
                (blank_heatmap_html)
            }
        }
        p {"This neuron has rank " (neuron_rank) " variance within the layer."}
    }
}

fn generate_navigation_links(
    num_layers: usize,
    num_neurons: usize,
    layer_index: usize,
    neuron_index: usize,
) -> Markup {
    let previous_neuron_link = if neuron_index > 0 {
        html! {
            a href={"N"({neuron_index-1})} {
                "Previous"
            }
        }
    } else if layer_index > 0 {
        html! {
            a href={"../L"({layer_index-1})"/N"({num_neurons-1})} {
                "Previous layer"
            }
        }
    } else {
        html! {}
    };

    let next_neuron_link = if neuron_index < num_neurons - 1 {
        html! {
            a href={"N"({neuron_index+1})} {
                "Next"
            }
        }
    } else if layer_index < num_layers - 1 {
        html! {
            a href={"../L"({layer_index+1})"/N0"} {
                "Next layer"
            }
        }
    } else {
        html! {}
    };

    html! {
        (previous_neuron_link)" - "(next_neuron_link)
    }
}

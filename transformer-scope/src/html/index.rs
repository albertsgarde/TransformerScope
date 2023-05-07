use maud::{html, Markup};
use ndarray::{ArrayView2, Axis};

use crate::Payload;

pub fn generate_index_page(payload: &Payload, site: bool) -> Markup {
    let ranked_neurons = payload.ranked_neurons();

    html!(
        head {
            meta charset="utf-8";
            title {"TransformerScope"}
            link rel="stylesheet" href="static/style.css"{};
        }
        body {
            h1 {"Welcome to TransformerScope!"}
            p { "This is a web app for visualizing various aspects of transformer models." }
            p {"Based on the original idea of the "
            a href="https://neuroscope.io/gelu-3l/0/314.html"{"Neuroscope"} " by Neel
            Nanda."}
        }
        (generate_ranked_neurons_table(ranked_neurons.view(), site))
    )
}

pub fn generate_ranked_neurons_table(ranked_neurons: ArrayView2<u32>, site: bool) -> Markup {
    let (num_layers, _num_neurons) = ranked_neurons.dim();
    html!(
        table {
            tr {
                th;
                @for layer_index in 0..num_layers {
                    th {(format!("Layer {layer_index}"))}
                }
            }
            @for (rank, neurons_of_rank) in ranked_neurons.axis_iter(Axis(1)).enumerate() {
                tr {
                    th {(rank)}
                    @for (layer_index, neuron_index) in neurons_of_rank.iter().enumerate() {
                        td{
                            a href={"L"(layer_index)"/N"(neuron_index)(if site {".html"} else {""})} {(neuron_index)}
                        }
                    }
                }
            }
        }
    )
}

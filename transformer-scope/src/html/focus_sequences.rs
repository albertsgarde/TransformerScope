use maud::{html, Markup};
use ndarray::ArrayView2;

use super::heatmap;

pub fn focus_sequences(activations: ArrayView2<f32>, step_names: ArrayView2<String>) -> Markup {
    assert_eq!(activations.shape(), step_names.shape());
    html! {
        table class="games" {
            tr {
                td class = "game_step_id";
                @for column_index in 0..activations.ncols() {
                    td class = "game_step_id" {
                        ({column_index + 1})
                    }
                }

            }
            @for (row_index, (activation_row, step_name_row)) in activations.rows().into_iter().zip(step_names.rows().into_iter()).enumerate() {
                tr {
                    td class = "game_step_id" {
                        ({row_index + 1})
                    }
                    @for (&activation, step_name) in activation_row.iter().zip(step_name_row.iter()) {
                        (board_cell(activation, step_name))
                    }
                }
            }
        }
    }
}

fn board_cell(activation: f32, step_name: &str) -> impl maud::Render {
    let color = heatmap::interpolate_color(activation * 10.);

    html! {
        td class="game_step" style={"background-color: rgb("(color[0])", "(color[1])", "(color[2])")"} {
            (step_name)
        }
    }
}

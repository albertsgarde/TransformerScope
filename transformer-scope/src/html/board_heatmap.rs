use maud::{html, Markup};
use ndarray::Array2;

const COLOR_POSITIVE: [f32; 3] = [69., 254., 152.];
const COLOR_ZERO: [f32; 3] = [0., 0., 0.];
const COLOR_NEGATIVE: [f32; 3] = [255., 0., 0.];

fn interpolate_color(value: f32) -> [u8; 3] {
    let interpolation_color = if value > 0. {
        COLOR_POSITIVE
    } else {
        COLOR_NEGATIVE
    };
    let interpolation_value = value.abs();

    let mut color = [0u8, 0, 0];
    for (color_index, color_value) in color.iter_mut().enumerate() {
        let color_value_float = COLOR_ZERO[color_index]
            + (interpolation_color[color_index] - COLOR_ZERO[color_index]) * interpolation_value;
        *color_value = color_value_float.round() as u8;
    }

    color
}

pub fn board_heatmap(values: &Array2<f32>) -> Markup {
    html! {
        table {
            @for (row_index, row) in values.rows().into_iter().enumerate() {
                tr {
                    @for (column_index, &value) in row.iter().enumerate() {
                        (board_cell(value, row_index, column_index))
                    }
                }
            }
        }
    }
}

fn board_cell(value: f32, row_index: usize, column_index: usize) -> impl maud::Render {
    let color = interpolate_color(value * 10.);

    html! {
        td style={"background-color: rgb("(color[0])", "(color[1])", "(color[2])")"} {
            (char::from(b'A' + row_index as u8))({column_index + 1})
        }
    }
}

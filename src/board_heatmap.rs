use maud::{html, Markup};
use ndarray::{Array1, Array2};

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
    let color_ours = Array1::from_shape_vec(3, vec![69., 254., 152.]).unwrap();
    let color_blank = Array1::from_shape_vec(3, vec![0., 0., 0.]).unwrap();
    let color_theirs = Array1::from_shape_vec(3, vec![255., 0., 0.]).unwrap();

    let interpolation_color = if value > 0. { color_ours } else { color_theirs };
    let interpolation_value = value.abs() * 10.;

    let color = &color_blank + (interpolation_color - &color_blank) * interpolation_value;

    let color_string = format!("rgb({:.0}, {:.0}, {:.0})", color[0], color[1], color[2]);

    html! {
        td style=(format!("background-color: {}", color_string)) {
            (format!("{}{}", char::from(b'A' + row_index as u8), column_index + 1))
        }
    }
}

use maud::{html, Markup};
use ndarray::{s, Ix2};
use serde::{Deserialize, Serialize};

use crate::{html::heatmap, Payload};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Element {
    Heatmap(String),
    Value(String),
}

impl Element {
    fn parse_inner(str: &str) -> Self {
        let str = str.trim();
        let (element_name, args_string) = str.split_once('(').unwrap();
        assert_eq!(args_string.chars().last().unwrap(), ')');
        let args_string = &args_string[..args_string.len() - 1];
        let mut arg_strings = args_string.split(',').map(|s| s.trim());

        match element_name {
            "heatmap" => {
                let heatmap_name = arg_strings.next().unwrap();
                assert_eq!(arg_strings.next(), None);
                Element::Heatmap(heatmap_name.to_string())
            }
            "value" => {
                let value_name = arg_strings.next().unwrap();
                assert_eq!(arg_strings.next(), None);
                Element::Value(value_name.to_string())
            }
            _ => panic!("Invalid element name: {element_name}"),
        }
    }

    pub fn parse<S: AsRef<str>>(str: S) -> Self {
        Element::parse_inner(str.as_ref())
    }

    pub fn generate(&self, payload: &Payload, layer_index: usize, neuron_index: usize) -> Markup {
        match self {
            Element::Heatmap(heatmap_name) => {
                let heatmap = payload.value(heatmap_name).unwrap().as_f32().unwrap();
                assert_eq!(
                    &heatmap.shape()[..2],
                    &[payload.num_layers(), payload.num_mlp_neurons()]
                );
                let heatmap = heatmap.slice(s![layer_index, neuron_index, .., ..]);
                heatmap::heatmap(heatmap)
            }
            Element::Value(value) => match payload.value(value).unwrap().view() {
                crate::data::value::ValueView::String(array) => {
                    let array = array.into_dimensionality::<Ix2>().unwrap();
                    assert_eq!(
                        array.dim(),
                        (payload.num_layers(), payload.num_mlp_neurons())
                    );
                    let value = &array[(layer_index, neuron_index)];
                    html! {
                        (value)
                    }
                }
                crate::data::value::ValueView::U32(array) => {
                    let array = array.into_dimensionality::<Ix2>().unwrap();
                    assert_eq!(
                        array.dim(),
                        (payload.num_layers(), payload.num_mlp_neurons())
                    );
                    let value = array[(layer_index, neuron_index)];
                    html! {
                        (value)
                    }
                }
                crate::data::value::ValueView::F32(array) => {
                    let array = array.into_dimensionality::<Ix2>().unwrap();
                    assert_eq!(
                        array.dim(),
                        (payload.num_layers(), payload.num_mlp_neurons())
                    );
                    let value = array[(layer_index, neuron_index)];
                    html! {
                        (value)
                    }
                }
            },
        }
    }
}

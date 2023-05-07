use maud::{html, Markup};
use ndarray::{s, Ix2};
use serde::{Deserialize, Serialize};

use crate::{
    html::{focus_sequences, heatmap},
    Payload,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Element {
    Heatmap(String),
    Value(String),
    FocusSequences {
        activations: String,
        step_names: String,
    },
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
            "focus_sequences" => {
                let activations_value_name = arg_strings.next().unwrap();
                let step_names_value_name = arg_strings.next().unwrap();
                assert_eq!(arg_strings.next(), None);
                Element::FocusSequences {
                    activations: activations_value_name.to_string(),
                    step_names: step_names_value_name.to_string(),
                }
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
            Element::FocusSequences {
                activations,
                step_names,
            } => {
                let activations = payload.value(activations).unwrap().as_f32().unwrap();
                let step_names = payload.value(step_names).unwrap().as_string().unwrap();
                assert_eq!(
                    &activations.shape()[..2],
                    &[payload.num_layers(), payload.num_mlp_neurons()]
                );
                assert_eq!(step_names.shape(), activations.shape());

                let activations = activations.slice(s![layer_index, neuron_index, .., ..]);
                let step_names = step_names.slice(s![layer_index, neuron_index, .., ..]);
                focus_sequences::focus_sequences(activations, step_names)
            }
        }
    }
}

use maud::{html, Markup};
use serde::{Deserialize, Serialize};

use crate::{html::board_heatmap, Payload};

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
                let heatmap = payload.get_table(layer_index, neuron_index, heatmap_name);
                board_heatmap::board_heatmap(heatmap)
            }
            Element::Value(value) => {
                let value = payload.get_scalar(layer_index, neuron_index, value);
                html! {
                    (value)
                }
            }
        }
    }
}

use std::collections::HashMap;

use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};

use super::{value, Value};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Values {
    pub global_values: HashMap<String, Value<value::Global>>,
    pub layer_values: HashMap<String, Value<value::Layer>>,
    pub neuron_values: HashMap<String, Value<value::Neuron>>,
}

impl Values {
    pub fn empty() -> Self {
        Self {
            global_values: HashMap::new(),
            layer_values: HashMap::new(),
            neuron_values: HashMap::new(),
        }
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.global_values.contains_key(key)
            || self.layer_values.contains_key(key)
            || self.neuron_values.contains_key(key)
    }

    pub fn get_table(
        &self,
        layer_index: usize,
        neuron_index: usize,
        key: &str,
    ) -> Option<ArrayView2<f32>> {
        if let Some(value) = self.neuron_values.get(key) {
            Some(
                value
                    .get_table(layer_index, neuron_index)
                    .unwrap_or_else(|| panic!("Value '{key}' is not a table.")),
            )
        } else if let Some(value) = self.layer_values.get(key) {
            Some(
                value
                    .get_table(layer_index, neuron_index)
                    .unwrap_or_else(|| panic!("Value '{key}' is not a table.")),
            )
        } else {
            self.global_values.get(key).map(|value| {
                value
                    .get_table(layer_index, neuron_index)
                    .unwrap_or_else(|| panic!("Value '{key}' is not a table."))
            })
        }
    }

    pub fn get_scalar(&self, layer_index: usize, neuron_index: usize, key: &str) -> Option<f32> {
        if let Some(value) = self.neuron_values.get(key) {
            Some(
                value
                    .get_scalar(layer_index, neuron_index)
                    .unwrap_or_else(|| panic!("Value '{key}' is not a scalar.")),
            )
        } else if let Some(value) = self.layer_values.get(key) {
            Some(
                value
                    .get_scalar(layer_index, neuron_index)
                    .unwrap_or_else(|| panic!("Value '{key}' is not a scalar.")),
            )
        } else {
            self.global_values.get(key).map(|value| {
                value
                    .get_scalar(layer_index, neuron_index)
                    .unwrap_or_else(|| panic!("Value '{key}' is not a scalar."))
            })
        }
    }
}

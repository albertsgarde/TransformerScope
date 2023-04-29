use std::{fs::File, path::Path};

use ndarray::{ArrayView2, ArrayView4};

use serde::{Deserialize, Serialize};

use crate::html::template::{NeuronTemplate, Value};

use super::values::Values;

#[derive(Clone, Serialize, Deserialize)]
pub struct Payload {
    num_layers: usize,
    num_mlp_neurons: usize,

    mlp_neuron_template: NeuronTemplate,

    values: Values,
}

impl Payload {
    pub fn new(
        num_layers: usize,
        num_mlp_neurons: usize,
        mlp_neuron_template: NeuronTemplate,
        values: Values,
    ) -> Self {
        Self {
            num_layers,
            num_mlp_neurons,
            mlp_neuron_template,
            values,
        }
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Self {
        bincode::deserialize_from(File::open(path.as_ref()).unwrap()).unwrap()
    }

    pub fn to_file<P: AsRef<Path>>(&self, path: P) {
        bincode::serialize_into(File::create(path.as_ref()).unwrap(), self).unwrap();
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn num_neurons(&self) -> usize {
        self.num_mlp_neurons
    }

    pub fn ownership_heatmaps(&self) -> ArrayView4<f32> {
        match &self.values.neuron_values["ownership_heatmap"] {
            Value::Table(array) => array.view(),
            _ => panic!("Ownership heatmap is the wrong type."),
        }
    }

    pub fn blank_heatmaps(&self) -> ArrayView4<f32> {
        match &self.values.neuron_values["blank_heatmap"] {
            Value::Table(array) => array.view(),
            _ => panic!("Blank heatmap is the wrong type."),
        }
    }

    pub fn ranked_neurons(&self) -> ArrayView2<f32> {
        self.get_table(0, 0, "ranked_neurons")
    }

    pub fn neuron_template(&self) -> &NeuronTemplate {
        &self.mlp_neuron_template
    }

    pub fn get_table(&self, layer_index: usize, neuron_index: usize, key: &str) -> ArrayView2<f32> {
        self.values
            .get_table(layer_index, neuron_index, key)
            .unwrap_or_else(|| panic!("Table '{key}' not found."))
    }

    pub fn get_scalar(&self, layer_index: usize, neuron_index: usize, key: &str) -> f32 {
        self.values
            .get_scalar(layer_index, neuron_index, key)
            .unwrap_or_else(|| panic!("Scalar '{key}' not found."))
    }
}

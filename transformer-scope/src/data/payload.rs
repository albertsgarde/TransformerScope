use std::path::Path;

use ndarray::{ArrayView2, Ix2};

use serde::{Deserialize, Serialize};

use crate::html::template::{ArgumentError, NeuronTemplate};

use super::{values::Values, Value};

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
    ) -> Result<Self, ArgumentError> {
        let result = Self {
            num_layers,
            num_mlp_neurons,
            mlp_neuron_template,
            values,
        };
        result
            .mlp_neuron_template
            .validate_arguments(&result)
            .map(|_| result)
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Self {
        let data = std::fs::read(path.as_ref()).unwrap();
        postcard::from_bytes(data.as_slice()).unwrap()
    }

    pub fn to_file<P: AsRef<Path>>(&self, path: P) {
        let data = postcard::to_allocvec(&self).unwrap();
        std::fs::write(path.as_ref(), data).unwrap();
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn num_mlp_neurons(&self) -> usize {
        self.num_mlp_neurons
    }

    pub fn ranked_neurons(&self) -> ArrayView2<u32> {
        self.values
            .get("ranked_neurons")
            .expect("Ranked neurons not set.")
            .as_u32()
            .unwrap()
            .view()
            .into_dimensionality::<Ix2>()
            .unwrap()
    }

    pub fn neuron_template(&self) -> &NeuronTemplate {
        &self.mlp_neuron_template
    }

    pub fn value(&self, key: impl AsRef<str>) -> Option<&Value> {
        self.values.get(key.as_ref())
    }
}

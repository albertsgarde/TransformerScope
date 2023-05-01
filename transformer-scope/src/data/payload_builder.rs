use std::collections::HashMap;

use crate::{html::template::NeuronTemplate, Payload};

use super::{neuron_rankings, value, values::Values, Value};

pub struct PayloadBuilder {
    num_layers: usize,
    num_mlp_neurons: usize,

    mlp_neuron_template: Option<NeuronTemplate>,
    global_values: HashMap<String, Value<value::Global>>,
    layer_values: HashMap<String, Value<value::Layer>>,
    neuron_values: HashMap<String, Value<value::Neuron>>,

    rank_values_key: Option<String>,
}

impl PayloadBuilder {
    pub fn new(num_layers: usize, num_mlp_neurons: usize) -> Self {
        Self {
            num_layers,
            num_mlp_neurons,
            mlp_neuron_template: None,
            global_values: HashMap::new(),
            layer_values: HashMap::new(),
            neuron_values: HashMap::new(),
            rank_values_key: None,
        }
    }

    pub fn mlp_neuron_template(&mut self, neuron_template: NeuronTemplate) {
        assert!(
            self.mlp_neuron_template.is_none(),
            "MLP neuron template already set."
        );
        self.mlp_neuron_template = Some(neuron_template);
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.global_values.contains_key(key)
            || self.layer_values.contains_key(key)
            || self.neuron_values.contains_key(key)
    }

    pub fn add_global_value(&mut self, key: impl Into<String>, value: Value<value::Global>) {
        let key: String = key.into();
        if self.contains_key(&key) {
            panic!("Value {key} already set.");
        }
        self.global_values.insert(key, value);
    }

    pub fn add_layer_value(&mut self, key: impl Into<String>, value: Value<value::Layer>) {
        let key: String = key.into();
        if self.contains_key(&key) {
            panic!("Value {key} already set.");
        }
        self.layer_values.insert(key, value);
    }

    pub fn add_neuron_value(&mut self, key: impl Into<String>, value: Value<value::Neuron>) {
        let key: String = key.into();
        if self.contains_key(&key) {
            panic!("Value {key} already set.");
        }
        self.neuron_values.insert(key, value);
    }

    pub fn set_rank_values(&mut self, key: impl Into<String>) {
        let key: String = key.into();
        if !self.contains_key(&key) {
            panic!("No value named {key} found. Please add the value before setting it as the rank value.");
        }
        if let Some(rank_values) = self.neuron_values.get(&key) {
            if rank_values.get_all_scalars().is_none() {
                panic!(
                    "Rank values must be a scalar value. '{key}' is a {} value.",
                    rank_values.type_string()
                );
            }
        } else {
            let locality_string = if self.global_values.contains_key(&key) {
                "global"
            } else {
                "layer"
            };
            panic!("Rank values must be a neuron value. '{key}' is a {locality_string} value.");
        }
        self.rank_values_key = Some(key);
    }

    pub fn build(self) -> Payload {
        if self.contains_key("rank") {
            panic!("Value name 'neuron_ranks' is reserved.")
        }
        if self.contains_key("ranked_neurons") {
            panic!("Value name 'ranked_neurons' is reserved.")
        }
        let Self {
            num_layers,
            num_mlp_neurons,
            mlp_neuron_template,
            mut global_values,
            layer_values,
            mut neuron_values,
            rank_values_key,
        } = self;

        if let Some(rank_values_key) = rank_values_key {
            let rank_values = neuron_values
                .get(&rank_values_key)
                .unwrap_or_else(|| panic!("Rank values key '{rank_values_key}' not found in `neuron_values`. This should be guaranteed by the `set_rank_values` method."))
                .get_all_scalars()
                .unwrap_or_else(|| panic!("Rank values '{rank_values_key}' not a scalar value. This should be guaranteed by the `set_rank_values` method."));
            let (neuron_ranks, ranked_neurons) =
                neuron_rankings::calculate_neuron_rankings(rank_values);
            let float_neuron_ranks = neuron_ranks.map(|&rank| rank as f32);
            let float_ranked_neurons = ranked_neurons.map(|&rank| rank as f32);
            neuron_values.insert("rank".to_string(), Value::Scalar(float_neuron_ranks));
            global_values.insert(
                "ranked_neurons".to_string(),
                Value::Table(float_ranked_neurons),
            );
        }

        let values = Values::new(global_values, layer_values, neuron_values);

        Payload::new(
            num_layers,
            num_mlp_neurons,
            mlp_neuron_template.unwrap(),
            values,
        )
    }
}

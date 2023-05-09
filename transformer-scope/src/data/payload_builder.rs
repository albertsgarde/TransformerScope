use std::collections::HashMap;

use ndarray::Ix2;

use crate::{html::template::NeuronTemplate, Payload};

use super::{neuron_rankings, value::{self, Scope}, values::Values, Value};

pub struct PayloadBuilder {
    num_layers: usize,
    num_mlp_neurons: usize,

    mlp_neuron_template: Option<NeuronTemplate>,
    values: HashMap<String, Value>,

    rank_values_key: Option<String>,
}

impl PayloadBuilder {
    pub fn new(num_layers: usize, num_mlp_neurons: usize) -> Self {
        Self {
            num_layers,
            num_mlp_neurons,
            mlp_neuron_template: None,
            values: HashMap::new(),
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
        self.values.contains_key(key)
    }

    pub fn add_value(&mut self, key: impl Into<String>, value: Value) {
        match value.scope() {
            Scope::Global => {}
            Scope::Layer => assert_eq!(value.shape()[0], self.num_layers, 
                    "The first dimension of a value with scope `Layer` must have size equal to the number of layers."),
            Scope::Neuron => assert_eq!(value.shape()[..2], [self.num_layers, self.num_mlp_neurons], 
                    "The first and second dimensions of a value with scope `Neuron` must match the number of layers and the number of MLP neurons per layer."),
        }

        let key: String = key.into();
        if self.contains_key(&key) {
            panic!("Value {key} already set.");
        }
        self.values.insert(key, value);
    }

    pub fn set_rank_values(&mut self, rank_values_key: impl Into<String>) {
        let key: String = rank_values_key.into();
        if let Some(rank_values) = self.values.get(&key) {
            assert_eq!(
                rank_values.shape(),
                &[self.num_layers, self.num_mlp_neurons],
                "Rank values must have shape [{}, {}], i.e. one element for every neuron.",
                self.num_layers,
                self.num_mlp_neurons,
            );
            assert_eq!(
                rank_values.data_type(),
                value::DataType::F32,
                "Rank values must have data type F32.",
            )
        } else {
            panic!("No value named {key} found. Please add the value before setting it as the rank value.");
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
            mut values,
            rank_values_key,
        } = self;

        if let Some(rank_values_key) = rank_values_key {
            let rank_values = values
                .get(&rank_values_key).unwrap_or_else(|| panic!("No value found with key '{rank_values_key}'. This should be guaranteed by the `set_rank_values` method."));
            let rank_values_type = rank_values.data_type();
            let rank_values = rank_values.as_f32().unwrap_or_else(|| panic!("Value with key '{rank_values_key}' has the data type '{rank_values_type}', but only F32 is supported. This should be guaranteed by teh `set_rank_values` method."));
            let rank_values = rank_values.view().into_dimensionality::<Ix2>().unwrap();
            assert_eq!(rank_values.shape(), &[num_layers, num_mlp_neurons]);

            let (neuron_ranks, ranked_neurons) =
                neuron_rankings::calculate_neuron_rankings(rank_values);
            values.insert(
                "rank".to_string(),
                Value::new(
                    neuron_ranks.map(|&x| u32::try_from(x).unwrap()),
                    Scope::Neuron,
                ),
            );
            values.insert(
                "ranked_neurons".to_string(),
                Value::new(
                    ranked_neurons.map(|&x| u32::try_from(x).unwrap()),
                    Scope::Global,
                ),
            );
        }

        let values = Values::new(values);

        Payload::new(
            num_layers,
            num_mlp_neurons,
            mlp_neuron_template.unwrap(),
            values,
        )
    }
}

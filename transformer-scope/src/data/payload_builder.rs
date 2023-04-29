use crate::{html::template::NeuronTemplate, Payload};

use super::{neuron_rankings, value, values::Values, Value};

pub struct PayloadBuilder {
    num_layers: usize,
    num_mlp_neurons: usize,

    mlp_neuron_template: Option<NeuronTemplate>,
    values: Values,
}

impl PayloadBuilder {
    pub fn new(num_layers: usize, num_mlp_neurons: usize) -> Self {
        Self {
            num_layers,
            num_mlp_neurons,
            mlp_neuron_template: None,
            values: Values::empty(),
        }
    }

    pub fn mlp_neuron_template(&mut self, neuron_template: NeuronTemplate) {
        assert!(
            self.mlp_neuron_template.is_none(),
            "MLP neuron template already set."
        );
        self.mlp_neuron_template = Some(neuron_template);
    }

    pub fn add_global_value(&mut self, key: impl Into<String>, value: Value<value::Global>) {
        let key: String = key.into();
        if self.values.contains_key(&key) {
            panic!("Value {key} already set.");
        }
        self.values.global_values.insert(key, value);
    }

    pub fn add_neuron_value(&mut self, key: impl Into<String>, value: Value<value::Neuron>) {
        let key: String = key.into();
        if self.values.contains_key(&key) {
            panic!("Value {key} already set.");
        }
        self.values.neuron_values.insert(key, value);
    }

    pub fn build(self) -> Payload {
        let Self {
            num_layers,
            num_mlp_neurons,
            mlp_neuron_template,
            mut values,
        } = self;
        if values.contains_key("rank") {
            panic!("Value name 'neuron_ranks' is reserved.")
        }
        if values.contains_key("ranked_neurons") {
            panic!("Value name 'ranked_neurons' is reserved.")
        }
        let ownership_heatmaps = match &values.neuron_values["ownership_heatmap"] {
            Value::Table(array) => array.view(),
            _ => panic!("Ownership heatmap is the wrong type."),
        };
        let (neuron_ranks, ranked_neurons) =
            neuron_rankings::calculate_neuron_rankings(ownership_heatmaps);
        let float_neuron_ranks = neuron_ranks.map(|&rank| rank as f32);
        let float_ranked_neurons = ranked_neurons.map(|&rank| rank as f32);
        values
            .neuron_values
            .insert("rank".to_string(), Value::Scalar(float_neuron_ranks));
        values.global_values.insert(
            "ranked_neurons".to_string(),
            Value::Table(float_ranked_neurons),
        );

        Payload::new(
            num_layers,
            num_mlp_neurons,
            mlp_neuron_template.unwrap(),
            values,
        )
    }
}

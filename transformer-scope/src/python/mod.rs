use numpy::borrow::{PyReadonlyArray2, PyReadonlyArray4};
use pyo3::prelude::*;

use crate::{
    data::{Payload, PayloadBuilder, Value},
    html::template::NeuronTemplate,
};

#[pyclass(name = "ValueType", frozen)]
enum ValueType {
    Table,
    Array,
    Scalar,
}

#[pymethods]
impl ValueType {
    #[classattr]
    fn table() -> Self {
        ValueType::Table
    }

    #[classattr]
    fn array() -> Self {
        ValueType::Array
    }

    #[classattr]
    fn scalar() -> Self {
        ValueType::Scalar
    }
}

#[pyclass(name = "PayloadBuilder")]
struct PyPayloadBuilder {
    payload_builder: Option<PayloadBuilder>,
}

impl PyPayloadBuilder {
    fn get(&mut self) -> &mut PayloadBuilder {
        self.payload_builder
            .as_mut()
            .expect("Payload already built!")
    }
}

#[pymethods]
impl PyPayloadBuilder {
    #[new]
    pub fn new(num_layers: usize, num_mlp_neurons: usize) -> Self {
        let payload_builder = Some(PayloadBuilder::new(num_layers, num_mlp_neurons));
        PyPayloadBuilder { payload_builder }
    }

    pub fn mlp_neuron_template(&mut self, neuron_template: &str) {
        let neuron_template = NeuronTemplate::parse(neuron_template);
        self.get().mlp_neuron_template(neuron_template);
    }

    pub fn add_global_table(&mut self, key: &str, value: PyReadonlyArray2<f32>) {
        self.get()
            .add_global_value(key, Value::Table(value.as_array().to_owned()));
    }

    pub fn add_neuron_table(&mut self, key: &str, value: PyReadonlyArray4<f32>) {
        self.get()
            .add_neuron_value(key, Value::Table(value.as_array().to_owned()));
    }

    pub fn add_neuron_scalar(&mut self, key: &str, value: PyReadonlyArray2<f32>) {
        self.get()
            .add_neuron_value(key, Value::Scalar(value.as_array().to_owned()));
    }

    pub fn set_rank_values(&mut self, key: &str) {
        self.get().set_rank_values(key);
    }

    pub fn build(&mut self) -> PyPayload {
        let payload = self
            .payload_builder
            .take()
            .unwrap_or_else(|| panic!("Payload already built!"))
            .build();
        PyPayload { payload }
    }
}

#[pyclass(name = "Payload", frozen)]
struct PyPayload {
    payload: Payload,
}

#[pymethods]
impl PyPayload {
    pub fn to_file(&self, path: &str) {
        self.payload.to_file(path);
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn transformer_scope(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPayloadBuilder>()?;
    m.add_class::<PyPayload>()?;
    Ok(())
}

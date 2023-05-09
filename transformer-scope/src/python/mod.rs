use ndarray::ArrayD;
use numpy::borrow::PyReadonlyArrayDyn;
use pyo3::{create_exception, exceptions::PyException, prelude::*};

use crate::{
    data::{value::Scope, Payload, PayloadBuilder, Value},
    html::template::{ArgumentError, NeuronTemplate},
};

create_exception!(transformer_scope, PayloadBuildError, PyException);

impl From<ArgumentError> for PyErr {
    fn from(value: ArgumentError) -> Self {
        PyErr::new::<PayloadBuildError, _>(format!("{value}"))
    }
}

#[pyclass(name = "Scope")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PyScope {
    Global,
    Layer,
    Neuron,
}

impl From<PyScope> for Scope {
    fn from(value: PyScope) -> Self {
        match value {
            PyScope::Global => Scope::Global,
            PyScope::Layer => Scope::Layer,
            PyScope::Neuron => Scope::Neuron,
        }
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

    pub fn add_str_value(
        &mut self,
        key: &str,
        value: PyReadonlyArrayDyn<PyObject>,
        scope: PyScope,
        py: Python<'_>,
    ) {
        let value_array: ArrayD<String> = value.as_array().map(|obj| obj.extract(py).unwrap());
        let value = Value::new(value_array, scope.into());
        self.get().add_value(key, value);
    }

    pub fn add_u32_value(&mut self, key: &str, value: PyReadonlyArrayDyn<u32>, scope: PyScope) {
        let value_array = value.as_array().to_owned();
        let value = Value::new(value_array, scope.into());
        self.get().add_value(key, value);
    }

    pub fn add_f32_value(&mut self, key: &str, value: PyReadonlyArrayDyn<f32>, scope: PyScope) {
        let value_array = value.as_array().to_owned();
        let value = Value::new(value_array, scope.into());
        self.get().add_value(key, value);
    }

    pub fn set_rank_values(&mut self, key: &str) {
        self.get().set_rank_values(key);
    }

    pub fn build(&mut self) -> Result<PyPayload, ArgumentError> {
        let payload = self
            .payload_builder
            .take()
            .unwrap_or_else(|| panic!("Payload already built!"))
            .build()?;
        Ok(PyPayload { payload })
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

    pub fn generate_site_files(&self, dir_path: &str) {
        crate::html::generate_site_in_dir(dir_path, &self.payload);
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn transformer_scope(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPayloadBuilder>()?;
    m.add_class::<PyPayload>()?;
    m.add_class::<PyScope>()?;
    m.add("PayloadBuildError", py.get_type::<PayloadBuildError>())?;

    Ok(())
}

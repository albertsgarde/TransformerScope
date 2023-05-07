use ndarray::ArrayD;
use numpy::borrow::PyReadonlyArrayDyn;
use pyo3::prelude::*;

use crate::{
    data::{Payload, PayloadBuilder},
    html::template::NeuronTemplate,
};

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

    pub fn add_string_value(
        &mut self,
        key: &str,
        value: PyReadonlyArrayDyn<PyObject>,
        py: Python<'_>,
    ) {
        assert_eq!(value.dtype().kind(), b'U', "Value must be a string.");
        let value: ArrayD<String> = value.as_array().map(|obj| obj.extract(py).unwrap());
        self.get().add_value(key, value.to_owned().into());
    }

    pub fn add_u32_value(&mut self, key: &str, value: PyReadonlyArrayDyn<u32>) {
        self.get()
            .add_value(key, value.as_array().to_owned().into());
    }

    pub fn add_f32_value(&mut self, key: &str, value: PyReadonlyArrayDyn<f32>) {
        self.get()
            .add_value(key, value.as_array().to_owned().into());
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

    pub fn generate_site_files(&self, dir_path: &str) {
        crate::html::generate_site_in_dir(dir_path, &self.payload);
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn transformer_scope(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPayloadBuilder>()?;
    m.add_class::<PyPayload>()?;
    Ok(())
}

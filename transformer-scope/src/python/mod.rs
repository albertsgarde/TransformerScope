use numpy::borrow::PyReadonlyArray4;
use pyo3::prelude::*;

use crate::{html::template::NeuronTemplate, Payload};

#[pyclass(name = "Payload")]
struct PyPayload {
    payload: Payload,
}

#[pymethods]
impl PyPayload {
    #[new]
    pub fn new(
        ownership_heatmaps: PyReadonlyArray4<f32>,
        blank_heatmaps: PyReadonlyArray4<f32>,
        neuron_template: &str,
    ) -> Self {
        let ownership_heatmaps = ownership_heatmaps.as_array().to_owned();
        let blank_heatmaps = blank_heatmaps.as_array().to_owned();

        let neuron_template = NeuronTemplate::parse(neuron_template);

        let payload = Payload::new(ownership_heatmaps, blank_heatmaps, neuron_template);
        PyPayload { payload }
    }

    pub fn to_dir(&self, path: &str) {
        self.payload.to_dir(path);
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn transformer_scope(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPayload>()?;
    Ok(())
}

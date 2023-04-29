use numpy::borrow::PyReadonlyArray4;
use pyo3::prelude::*;

use crate::Payload;

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
    ) -> Self {
        let ownership_heatmaps = ownership_heatmaps.as_array().to_owned();
        let blank_heatmaps = blank_heatmaps.as_array().to_owned();
        let payload = Payload::new(ownership_heatmaps, blank_heatmaps);
        PyPayload { payload }
    }

    pub fn to_dir(&self, path: &str) {
        self.payload.to_dir(path);
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn transformer_scope(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyPayload>()?;
    Ok(())
}

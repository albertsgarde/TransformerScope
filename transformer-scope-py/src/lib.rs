use pyo3::prelude::*;
mod python;

/// A Python module implemented in Rust.
#[pymodule]
fn transformer_scope_py(py: Python, m: &PyModule) -> PyResult<()> {
    python::module(py, m)
}

use thiserror::Error;

mod element;
pub use element::Element;
mod neuron_template;
pub use neuron_template::NeuronTemplate;

use crate::data::value::{DataType, Scope};

#[derive(Clone, Debug, Error)]
pub enum ArgumentErrorType {
    #[error("Argument with name {0} does not exist in payload.")]
    MissingValue(String),
    #[error("Argument has wrong data type. Required datatype is {required_data_type} but found datatype {found_data_type}.")]
    DataType {
        required_data_type: DataType,
        found_data_type: DataType,
    },
    #[error("Argument has wrong scope. Required scope is {required_scope} but found scope {found_scope}.")]
    Scope {
        required_scope: Scope,
        found_scope: Scope,
    },
    #[error("Argument has wrong number of axes. Required number is {required_axis_num} but found {found_axis_num}. Perhaps the scope is incorrect?")]
    AxisNum {
        required_axis_num: usize,
        found_axis_num: usize,
    },
    #[error("{0}")]
    Other(String),
}

#[derive(Clone, Debug, Error)]
#[error("{error_type}")]
pub struct ArgumentError {
    error_type: ArgumentErrorType,
    value_name: String,
}

use maud::{html, Markup};
use ndarray::{s, Ix2};
use serde::{Deserialize, Serialize};

use crate::{
    data::value::{DataType, Scope},
    html::{focus_sequences, heatmap},
    Payload,
};

use super::{ArgumentError, ArgumentErrorType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Element {
    Heatmap(String),
    Value(String),
    FocusSequences {
        activations: String,
        step_names: String,
    },
}

impl Element {
    fn parse_inner(str: &str) -> Self {
        let str = str.trim();
        let (element_name, args_string) = str.split_once('(').unwrap();
        assert_eq!(args_string.chars().last().unwrap(), ')');
        let args_string = &args_string[..args_string.len() - 1];
        let mut arg_strings = args_string.split(',').map(|s| s.trim());

        match element_name {
            "heatmap" => {
                let heatmap_name = arg_strings.next().unwrap();
                assert_eq!(arg_strings.next(), None);
                Element::Heatmap(heatmap_name.to_string())
            }
            "value" => {
                let value_name = arg_strings.next().unwrap();
                assert_eq!(arg_strings.next(), None);
                Element::Value(value_name.to_string())
            }
            "focus_sequences" => {
                let activations_value_name = arg_strings.next().unwrap();
                let step_names_value_name = arg_strings.next().unwrap();
                assert_eq!(arg_strings.next(), None);
                Element::FocusSequences {
                    activations: activations_value_name.to_string(),
                    step_names: step_names_value_name.to_string(),
                }
            }
            _ => panic!("Invalid element name: {element_name}"),
        }
    }

    pub fn parse<S: AsRef<str>>(str: S) -> Self {
        Element::parse_inner(str.as_ref())
    }

    pub fn generate(&self, payload: &Payload, layer_index: usize, neuron_index: usize) -> Markup {
        match self {
            Element::Heatmap(heatmap_name) => {
                let heatmap = payload.value(heatmap_name).unwrap().as_f32().unwrap();
                assert_eq!(
                    &heatmap.shape()[..2],
                    &[payload.num_layers(), payload.num_mlp_neurons()]
                );
                let heatmap = heatmap.slice(s![layer_index, neuron_index, .., ..]);
                heatmap::heatmap(heatmap)
            }
            Element::Value(value) => match payload.value(value).unwrap().view() {
                crate::data::value::ValueView::String(array) => {
                    let array = array.into_dimensionality::<Ix2>().unwrap();
                    assert_eq!(
                        array.dim(),
                        (payload.num_layers(), payload.num_mlp_neurons())
                    );
                    let value = &array[(layer_index, neuron_index)];
                    html! {
                        (value)
                    }
                }
                crate::data::value::ValueView::U32(array) => {
                    let array = array.into_dimensionality::<Ix2>().unwrap();
                    assert_eq!(
                        array.dim(),
                        (payload.num_layers(), payload.num_mlp_neurons())
                    );
                    let value = array[(layer_index, neuron_index)];
                    html! {
                        (value)
                    }
                }
                crate::data::value::ValueView::F32(array) => {
                    let array = array.into_dimensionality::<Ix2>().unwrap();
                    assert_eq!(
                        array.dim(),
                        (payload.num_layers(), payload.num_mlp_neurons())
                    );
                    let value = array[(layer_index, neuron_index)];
                    html! {
                        (value)
                    }
                }
            },
            Element::FocusSequences {
                activations,
                step_names,
            } => {
                let activations_value = payload.value(activations).unwrap();
                let step_names_value = payload.value(step_names).unwrap();

                let activations = activations_value.as_f32().unwrap();
                let step_names = step_names_value.as_string().unwrap();
                assert_eq!(
                    &activations.shape()[..2],
                    &[payload.num_layers(), payload.num_mlp_neurons()]
                );

                let activations = activations.slice(s![layer_index, neuron_index, .., ..]);
                let step_names = match step_names_value.scope() {
                    Scope::Global => step_names.view().into_dimensionality().unwrap(),
                    Scope::Layer => step_names.slice(s![layer_index, .., ..]),
                    Scope::Neuron => step_names.slice(s![layer_index, neuron_index, .., ..]),
                };
                assert_eq!(activations.shape(), step_names.shape());
                focus_sequences::focus_sequences(activations, step_names)
            }
        }
    }

    pub fn validate_arguments(&self, payload: &Payload) -> Result<(), ArgumentError> {
        match self {
            Element::Heatmap(heatmap_key) => payload
                .value(heatmap_key)
                .ok_or_else(|| ArgumentErrorType::MissingValue(heatmap_key.clone()))
                .and_then(|heatmap| {
                    let heatmap_axis_num = match heatmap.scope() {
                        Scope::Global => heatmap.shape().len(),
                        Scope::Layer => heatmap.shape().len() - 1,
                        Scope::Neuron => heatmap.shape().len() - 2,
                    };
                    if heatmap_axis_num != 2 {
                        Err(ArgumentErrorType::AxisNum {
                            required_axis_num: 2,
                            found_axis_num: heatmap_axis_num,
                        })
                    } else if heatmap.data_type() != DataType::F32 {
                        Err(ArgumentErrorType::DataType {
                            required_data_type: DataType::F32,
                            found_data_type: heatmap.data_type(),
                        })
                    } else {
                        Ok(())
                    }
                })
                .map_err(|error_type| ArgumentError {
                    error_type,
                    value_name: heatmap_key.to_owned(),
                }),
            Element::Value(value_key) => payload
                .value(value_key)
                .ok_or_else(|| ArgumentErrorType::MissingValue(value_key.clone()))
                .and_then(|value| {
                    let value_axis_num = match value.scope() {
                        Scope::Global => value.shape().len(),
                        Scope::Layer => value.shape().len() - 1,
                        Scope::Neuron => value.shape().len() - 2,
                    };
                    if value_axis_num != 0 {
                        Err(ArgumentErrorType::AxisNum {
                            required_axis_num: 0,
                            found_axis_num: value_axis_num,
                        })
                    } else {
                        Ok(())
                    }
                })
                .map_err(|error_type| ArgumentError {
                    error_type,
                    value_name: value_key.to_owned(),
                }),
            Element::FocusSequences {
                activations: activations_key,
                step_names: step_names_key,
            } => {
                let activations = payload
                    .value(activations_key)
                    .ok_or_else(|| ArgumentErrorType::MissingValue(activations_key.clone()))
                    .and_then(|activations| {
                        let activations_axis_num = match activations.scope() {
                            Scope::Global => activations.shape().len(),
                            Scope::Layer => activations.shape().len() - 1,
                            Scope::Neuron => activations.shape().len() - 2,
                        };
                        if activations_axis_num != 2 {
                            Err(ArgumentErrorType::AxisNum {
                                required_axis_num: 2,
                                found_axis_num: activations_axis_num,
                            })
                        } else if activations.data_type() != DataType::F32 {
                            Err(ArgumentErrorType::DataType {
                                required_data_type: DataType::F32,
                                found_data_type: activations.data_type(),
                            })
                        } else {
                            Ok(activations)
                        }
                    })
                    .map_err(|error_type: ArgumentErrorType| ArgumentError {
                        error_type,
                        value_name: activations_key.to_owned(),
                    })?;

                payload
                    .value(step_names_key)
                    .ok_or_else(|| ArgumentErrorType::MissingValue(step_names_key.clone())).and_then(|step_names| {
                        let step_names_shape = match step_names.scope() {
                            Scope::Global => step_names.shape(),
                            Scope::Layer => &step_names.shape()[1..],
                            Scope::Neuron => &step_names.shape()[2..],
                        };

                        let activations_shape = match activations.scope() {
                            Scope::Global => activations.shape(),
                            Scope::Layer => &activations.shape()[1..],
                            Scope::Neuron => &activations.shape()[2..],
                        };

                        if step_names_shape.len() != 2 {
                            Err(ArgumentErrorType::AxisNum {
                                required_axis_num: 2,
                                found_axis_num: step_names_shape.len(),
                            })
                        } else if step_names_shape != activations_shape {
                            Err(ArgumentErrorType::Other(format!("The two arguments to the element 'focus_sequences' must have equal shape (after scope). \
                                    First argument has shape {:?} while second argument has shape {:?}.",
                                    activations_shape, step_names_shape)))
                        } else if step_names.data_type() != DataType::String {
                            Err(ArgumentErrorType::DataType {
                                required_data_type: DataType::String,
                                found_data_type: step_names.data_type(),
                            })
                        } else {
                            Ok(())
                        }
                    })
                    .map_err(|error_type: ArgumentErrorType| ArgumentError {
                        error_type,
                        value_name: step_names_key.to_owned(),
                    })
            }
        }
    }
}

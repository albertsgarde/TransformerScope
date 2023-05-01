use ndarray::{s, Array, ArrayView, ArrayView1, ArrayView2, Dimension, RemoveAxis};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::fmt::Debug;
pub trait ValueLocality: Clone + Copy + Debug + Serialize + DeserializeOwned {
    type D0: Dimension + Serialize + DeserializeOwned;
    type D1: Dimension + RemoveAxis + Serialize + DeserializeOwned;
    type D2: Dimension + RemoveAxis + Serialize + DeserializeOwned;

    fn get_neuron_0d(array: &Array<f32, Self::D0>, layer_index: usize, neuron_index: usize) -> f32;
    fn get_neuron_1d(
        array: &Array<f32, Self::D1>,
        layer_index: usize,
        neuron_index: usize,
    ) -> ArrayView1<f32>;
    fn get_neuron_2d(
        array: &Array<f32, Self::D2>,
        layer_index: usize,
        neuron_index: usize,
    ) -> ArrayView2<f32>;
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Global;
impl ValueLocality for Global {
    type D0 = ndarray::Ix0;
    type D1 = ndarray::Ix1;
    type D2 = ndarray::Ix2;

    fn get_neuron_0d(
        array: &Array<f32, Self::D0>,
        _layer_index: usize,
        _neuron_index: usize,
    ) -> f32 {
        *array.get(()).unwrap()
    }

    fn get_neuron_1d(
        array: &Array<f32, Self::D1>,
        _layer_index: usize,
        _neuron_index: usize,
    ) -> ArrayView1<f32> {
        array.view()
    }

    fn get_neuron_2d(
        array: &Array<f32, Self::D2>,
        _layer_index: usize,
        _neuron_index: usize,
    ) -> ArrayView2<f32> {
        array.view()
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Layer;
impl ValueLocality for Layer {
    type D0 = ndarray::Ix1;
    type D1 = ndarray::Ix2;
    type D2 = ndarray::Ix3;

    fn get_neuron_0d(
        array: &Array<f32, Self::D0>,
        layer_index: usize,
        _neuron_index: usize,
    ) -> f32 {
        *array.get([layer_index]).unwrap()
    }

    fn get_neuron_1d(
        array: &Array<f32, Self::D1>,
        layer_index: usize,
        _neuron_index: usize,
    ) -> ArrayView1<f32> {
        array.slice(s![layer_index, ..])
    }

    fn get_neuron_2d(
        array: &Array<f32, Self::D2>,
        layer_index: usize,
        _neuron_index: usize,
    ) -> ArrayView2<f32> {
        array.slice(s![layer_index, .., ..])
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Neuron;
impl ValueLocality for Neuron {
    type D0 = ndarray::Ix2;
    type D1 = ndarray::Ix3;
    type D2 = ndarray::Ix4;

    fn get_neuron_0d(array: &Array<f32, Self::D0>, layer_index: usize, neuron_index: usize) -> f32 {
        *array.get([layer_index, neuron_index]).unwrap()
    }

    fn get_neuron_1d(
        array: &Array<f32, Self::D1>,
        layer_index: usize,
        neuron_index: usize,
    ) -> ArrayView1<f32> {
        array.slice(s![layer_index, neuron_index, ..])
    }

    fn get_neuron_2d(
        array: &Array<f32, Self::D2>,
        layer_index: usize,
        neuron_index: usize,
    ) -> ArrayView2<f32> {
        array.slice(s![layer_index, neuron_index, .., ..])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Value<L: ValueLocality> {
    Table(Array<f32, L::D2>),
    Array(Array<f32, L::D1>),
    Scalar(Array<f32, L::D0>),
}

impl<L: ValueLocality> Value<L> {
    pub fn get_all_tables(&self) -> Option<ArrayView<f32, L::D2>> {
        match self {
            Value::Table(array) => Some(array.view()),
            _ => None,
        }
    }
    pub fn get_all_arrays(&self) -> Option<ArrayView<f32, L::D1>> {
        match self {
            Value::Array(array) => Some(array.view()),
            _ => None,
        }
    }
    pub fn get_all_scalars(&self) -> Option<ArrayView<f32, L::D0>> {
        match self {
            Value::Scalar(array) => Some(array.view()),
            _ => None,
        }
    }

    pub fn get_table(&self, layer_index: usize, neuron_index: usize) -> Option<ArrayView2<f32>> {
        match self {
            Value::Table(array) => Some(L::get_neuron_2d(array, layer_index, neuron_index)),
            _ => None,
        }
    }

    pub fn get_array(&self, layer_index: usize, neuron_index: usize) -> Option<ArrayView1<f32>> {
        match self {
            Value::Array(array) => Some(L::get_neuron_1d(array, layer_index, neuron_index)),
            _ => None,
        }
    }

    pub fn get_scalar(&self, layer_index: usize, neuron_index: usize) -> Option<f32> {
        match self {
            Value::Scalar(array) => Some(L::get_neuron_0d(array, layer_index, neuron_index)),
            _ => None,
        }
    }

    pub fn type_string(&self) -> &'static str {
        match self {
            Value::Table(_) => "table",
            Value::Array(_) => "array",
            Value::Scalar(_) => "scalar",
        }
    }
}

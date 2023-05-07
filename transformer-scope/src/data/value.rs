use std::fmt::Display;

use delegate::delegate;
use ndarray::{Array, ArrayD, ArrayViewD};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataType {
    String,
    U32,
    F32,
}

impl Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ValueArray {
    String(ArrayD<String>),
    U32(ArrayD<u32>),
    F32(ArrayD<f32>),
}

pub enum ValueView<'a> {
    String(ArrayViewD<'a, String>),
    U32(ArrayViewD<'a, u32>),
    F32(ArrayViewD<'a, f32>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Value {
    array: ValueArray,
}

impl Value {
    pub fn from_string<D>(array: Array<String, D>) -> Self
    where
        D: ndarray::Dimension,
    {
        Self {
            array: ValueArray::String(array.into_dyn()),
        }
    }

    pub fn from_u32<D>(array: Array<u32, D>) -> Self
    where
        D: ndarray::Dimension,
    {
        Self {
            array: ValueArray::U32(array.into_dyn()),
        }
    }

    pub fn from_f32<D>(array: Array<f32, D>) -> Self
    where
        D: ndarray::Dimension,
    {
        Self {
            array: ValueArray::F32(array.into_dyn()),
        }
    }

    pub fn data_type(&self) -> DataType {
        match self.array {
            ValueArray::String(_) => DataType::String,
            ValueArray::U32(_) => DataType::U32,
            ValueArray::F32(_) => DataType::F32,
        }
    }

    pub fn view(&self) -> ValueView {
        match self.array {
            ValueArray::String(ref array) => ValueView::String(array.view()),
            ValueArray::U32(ref array) => ValueView::U32(array.view()),
            ValueArray::F32(ref array) => ValueView::F32(array.view()),
        }
    }

    delegate! {
        to match self.array {
            ValueArray::String(ref array) => array,
            ValueArray::U32(ref array) => array,
            ValueArray::F32(ref array) => array,
        } {
            pub fn shape(&self) -> &[usize];
        }
    }

    pub fn as_string(&self) -> Option<&ArrayD<String>> {
        match self.array {
            ValueArray::String(ref array) => Some(array),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<&ArrayD<u32>> {
        match self.array {
            ValueArray::U32(ref array) => Some(array),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<&ArrayD<f32>> {
        match self.array {
            ValueArray::F32(ref array) => Some(array),
            _ => None,
        }
    }
}

impl From<ArrayD<String>> for Value {
    fn from(array: ArrayD<String>) -> Self {
        Self::from_string(array)
    }
}

impl From<ArrayD<u32>> for Value {
    fn from(array: ArrayD<u32>) -> Self {
        Self::from_u32(array)
    }
}

impl From<ArrayD<f32>> for Value {
    fn from(array: ArrayD<f32>) -> Self {
        Self::from_f32(array)
    }
}

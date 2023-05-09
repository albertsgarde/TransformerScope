use std::fmt::Display;

use delegate::delegate;
use ndarray::{Array, ArrayD, ArrayViewD, Dimension};
use serde::{Deserialize, Serialize};

use private::ValueArray;

mod private {
    use ndarray::ArrayD;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum ValueArray {
        String(ArrayD<String>),
        U32(ArrayD<u32>),
        F32(ArrayD<f32>),
    }

    pub trait Data: Sized {
        fn to_value_array(array: ArrayD<Self>) -> ValueArray;
    }

    impl Data for String {
        fn to_value_array(array: ArrayD<Self>) -> ValueArray {
            ValueArray::String(array)
        }
    }

    impl Data for u32 {
        fn to_value_array(array: ArrayD<Self>) -> ValueArray {
            ValueArray::U32(array)
        }
    }

    impl Data for f32 {
        fn to_value_array(array: ArrayD<Self>) -> ValueArray {
            ValueArray::F32(array)
        }
    }
}

pub trait Data: private::Data {}

impl<T> Data for T where T: private::Data {}

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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Scope {
    Global,
    Layer,
    Neuron,
}

impl Display for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub enum ValueView<'a> {
    String(ArrayViewD<'a, String>),
    U32(ArrayViewD<'a, u32>),
    F32(ArrayViewD<'a, f32>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Value {
    array: ValueArray,
    scope: Scope,
}

impl Value {
    pub fn new<A, D>(array: Array<A, D>, scope: Scope) -> Self
    where
        A: Data,
        D: Dimension,
    {
        let array = A::to_value_array(array.into_dyn());
        Self { array, scope }
    }

    pub fn data_type(&self) -> DataType {
        match self.array {
            ValueArray::String(_) => DataType::String,
            ValueArray::U32(_) => DataType::U32,
            ValueArray::F32(_) => DataType::F32,
        }
    }

    pub fn scope(&self) -> Scope {
        self.scope
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

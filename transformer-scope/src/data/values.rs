use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::Value;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Values {
    values: HashMap<String, Value>,
}

impl Values {
    pub(super) fn new(values: HashMap<String, Value>) -> Self {
        Self { values }
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.values.get(key)
    }
}

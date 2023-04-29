use std::path::Path;

use crate::Payload;

#[derive(Clone)]
pub struct ApplicationState {
    payload: Payload,
}

impl ApplicationState {
    pub fn new<P>(path: P) -> Self
    where
        P: AsRef<Path>,
    {
        fn inner(path: &Path) -> ApplicationState {
            let payload = Payload::from_file(path);
            ApplicationState { payload }
        }
        inner(path.as_ref())
    }

    pub fn payload(&self) -> &Payload {
        &self.payload
    }
}

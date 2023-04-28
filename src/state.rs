use std::{fs::File, path::Path};

use ndarray::Array4;
use ndarray_npy::ReadNpyExt;

#[derive(Clone)]
pub struct ApplicationState {
    ownership_heatmaps: Array4<f32>,
}

impl ApplicationState {
    pub fn new<P>(path: P) -> Self
    where
        P: AsRef<Path>,
    {
        fn inner(path: &Path) -> ApplicationState {
            let file = File::open(path).unwrap();
            let ownership_heatmaps = Array4::<f32>::read_npy(file).unwrap();
            ApplicationState { ownership_heatmaps }
        }
        inner(path.as_ref())
    }

    pub fn ownership_heatmaps(&self) -> &Array4<f32> {
        &self.ownership_heatmaps
    }
}

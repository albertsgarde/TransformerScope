use std::{
    fs::{self, File},
    path::Path,
};

use ndarray::{Array2, Array4};
use ndarray_npy::{ReadNpyExt, WriteNpyExt};

use crate::data::neuron_rankings;

#[derive(Clone)]
pub struct Payload {
    ownership_heatmaps: Array4<f32>,
    neuron_ranks: Array2<usize>,
    ranked_neurons: Array2<usize>,
}

impl Payload {
    pub fn new(ownership_heatmaps: Array4<f32>) -> Self {
        let (neuron_ranks, ranked_neurons) =
            neuron_rankings::calculate_neuron_rankings(&ownership_heatmaps);
        Payload {
            ownership_heatmaps,
            neuron_ranks,
            ranked_neurons,
        }
    }

    pub fn from_dir<P: AsRef<Path>>(path: P) -> Self {
        fn inner(path: &Path) -> Payload {
            let ownership_heatmap_path = path.join("ownership_heatmaps.npy");

            let ownership_heatmap_file = File::open(ownership_heatmap_path).unwrap();
            let ownership_heatmaps = Array4::<f32>::read_npy(ownership_heatmap_file).unwrap();

            let (neuron_ranks, ranked_neurons) =
                neuron_rankings::calculate_neuron_rankings(&ownership_heatmaps);

            Payload {
                ownership_heatmaps,
                neuron_ranks,
                ranked_neurons,
            }
        }
        inner(path.as_ref())
    }

    fn to_dir_inner(&self, path: &Path) {
        fs::create_dir_all(path).unwrap();
        let ownership_heatmap_path = path.join("ownership_heatmaps.npy");
        let ownership_heatmap_file = File::create(ownership_heatmap_path).unwrap();
        self.ownership_heatmaps
            .write_npy(ownership_heatmap_file)
            .unwrap();
    }

    pub fn to_dir<P: AsRef<Path>>(&self, path: P) {
        self.to_dir_inner(path.as_ref())
    }

    pub fn ownership_heatmaps(&self) -> &Array4<f32> {
        &self.ownership_heatmaps
    }

    pub fn neuron_ranks(&self) -> &Array2<usize> {
        &self.neuron_ranks
    }

    pub fn ranked_neurons(&self) -> &Array2<usize> {
        &self.ranked_neurons
    }
}

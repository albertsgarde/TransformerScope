use std::{fs::File, path::Path};

use ndarray::{s, Array2, Array4, Axis};
use ndarray_npy::ReadNpyExt;

#[derive(Clone)]
pub struct Payload {
    ownership_heatmaps: Array4<f32>,
    neuron_ranks: Array2<usize>,
    ranked_neurons: Array2<usize>,
}

impl Payload {
    fn calculate_neuron_rankings(
        ownership_heatmaps: &Array4<f32>,
    ) -> (Array2<usize>, Array2<usize>) {
        let (num_layers, num_neurons, num_rows, num_cols) = ownership_heatmaps.dim();
        assert_eq!(num_rows, 8);
        assert_eq!(num_cols, 8);
        let heatmaps = ownership_heatmaps
            .slice(s![.., .., .., ..])
            .into_shape((num_layers, num_neurons, 64))
            .unwrap();
        let heatmap_stds = heatmaps.std_axis(Axis(2), 0.);
        let std_shape = heatmap_stds.dim();
        assert_eq!(std_shape.0, num_layers);
        assert_eq!(std_shape.1, num_neurons);

        let mut rankings = Array2::zeros((num_layers, num_neurons));
        let mut ranked_neurons = Array2::zeros((num_layers, num_neurons));

        for ((layer, mut rankings_layer), mut ranked_neurons_layer) in heatmap_stds
            .axis_iter(Axis(0))
            .zip(rankings.axis_iter_mut(Axis(0)))
            .zip(ranked_neurons.axis_iter_mut(Axis(0)))
        {
            let mut layer_vec: Vec<_> = layer.iter().enumerate().collect();
            layer_vec.sort_by(|(_, sd1), (_, sd2)| sd1.total_cmp(sd2));
            for (rank, (neuron_index, _)) in layer_vec.into_iter().enumerate() {
                rankings_layer[neuron_index] = rank;
                ranked_neurons_layer[rank] = neuron_index;
            }
        }
        (rankings, ranked_neurons)
    }

    pub fn from_dir<P>(path: P) -> Self
    where
        P: AsRef<Path>,
    {
        fn inner(path: &Path) -> Payload {
            let ownership_heatmap_path = path.join("ownership_heatmaps.npy");

            let ownership_heatmap_file = File::open(ownership_heatmap_path).unwrap();
            let ownership_heatmaps = Array4::<f32>::read_npy(ownership_heatmap_file).unwrap();

            let (neuron_ranks, ranked_neurons) =
                Payload::calculate_neuron_rankings(&ownership_heatmaps);

            Payload {
                ownership_heatmaps,
                neuron_ranks,
                ranked_neurons,
            }
        }
        inner(path.as_ref())
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

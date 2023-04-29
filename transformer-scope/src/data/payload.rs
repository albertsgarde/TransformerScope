use std::{
    fs::{self, File},
    path::Path,
};

use ndarray::{s, Array2, Array4, ArrayView2};
use ndarray_npy::{ReadNpyExt, WriteNpyExt};

use crate::{data::neuron_rankings, html::template::NeuronTemplate};

#[derive(Clone)]
pub struct Payload {
    ownership_heatmaps: Array4<f32>,
    blank_heatmaps: Array4<f32>,
    neuron_ranks: Array2<usize>,
    ranked_neurons: Array2<usize>,

    mlp_neuron_template: NeuronTemplate,
}

impl Payload {
    pub fn new(
        ownership_heatmaps: Array4<f32>,
        blank_heatmaps: Array4<f32>,
        mlp_neuron_template: NeuronTemplate,
    ) -> Self {
        assert_eq!(ownership_heatmaps.dim(), blank_heatmaps.dim());

        let (neuron_ranks, ranked_neurons) =
            neuron_rankings::calculate_neuron_rankings(&ownership_heatmaps);
        Payload {
            ownership_heatmaps,
            blank_heatmaps,
            neuron_ranks,
            ranked_neurons,
            mlp_neuron_template,
        }
    }

    pub fn from_dir<P: AsRef<Path>>(path: P) -> Self {
        fn inner(path: &Path) -> Payload {
            let ownership_heatmap_path = path.join("ownership_heatmaps.npy");

            let ownership_heatmap_file = File::open(ownership_heatmap_path).unwrap();
            let ownership_heatmaps = Array4::<f32>::read_npy(ownership_heatmap_file).unwrap();

            let blank_heatmap_path = path.join("blank_heatmaps.npy");

            let blank_heatmap_file = File::open(blank_heatmap_path).unwrap();
            let blank_heatmaps = Array4::<f32>::read_npy(blank_heatmap_file).unwrap();

            let mlp_neuron_template_path = path.join("mlp_neuron_template.html");
            let mlp_neuron_template =
                serde_json::from_reader(File::open(mlp_neuron_template_path).unwrap()).unwrap();

            Payload::new(ownership_heatmaps, blank_heatmaps, mlp_neuron_template)
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
        let blank_heatmap_path = path.join("blank_heatmaps.npy");
        let blank_heatmap_file = File::create(blank_heatmap_path).unwrap();
        self.blank_heatmaps.write_npy(blank_heatmap_file).unwrap();

        let mlp_neuron_template_path = path.join("mlp_neuron_template.html");
        serde_json::to_writer(
            File::create(mlp_neuron_template_path).unwrap(),
            &self.mlp_neuron_template,
        )
        .unwrap();
    }

    pub fn to_dir<P: AsRef<Path>>(&self, path: P) {
        self.to_dir_inner(path.as_ref())
    }

    pub fn num_layers(&self) -> usize {
        self.ownership_heatmaps.dim().0
    }

    pub fn num_neurons(&self) -> usize {
        self.ownership_heatmaps.dim().1
    }

    pub fn ownership_heatmaps(&self) -> &Array4<f32> {
        &self.ownership_heatmaps
    }

    pub fn blank_heatmaps(&self) -> &Array4<f32> {
        &self.blank_heatmaps
    }

    pub fn neuron_ranks(&self) -> &Array2<usize> {
        &self.neuron_ranks
    }

    pub fn ranked_neurons(&self) -> &Array2<usize> {
        &self.ranked_neurons
    }

    pub fn neuron_template(&self) -> &NeuronTemplate {
        &self.mlp_neuron_template
    }

    pub fn get_table(
        &self,
        layer_index: usize,
        neuron_index: usize,
        table_name: &str,
    ) -> ArrayView2<f32> {
        match table_name {
            "ownership_heatmap" => {
                self.ownership_heatmaps
                    .slice(s![layer_index, neuron_index, .., ..])
            }
            "blank_heatmap" => self
                .blank_heatmaps
                .slice(s![layer_index, neuron_index, .., ..]),
            _ => panic!("Unknown table name: {table_name}"),
        }
    }

    pub fn get_formatted_value(
        &self,
        layer_index: usize,
        neuron_index: usize,
        value_name: &str,
    ) -> String {
        match value_name {
            "rank" => self.neuron_ranks[[layer_index, neuron_index]].to_string(),
            _ => panic!("Unknown value name: {value_name}"),
        }
    }
}

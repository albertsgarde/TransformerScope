use ndarray::{Array2, ArrayView2, Axis};

pub fn calculate_neuron_rankings(
    ranking_values: ArrayView2<f32>,
) -> (Array2<usize>, Array2<usize>) {
    let (num_layers, num_neurons) = ranking_values.dim();

    let mut rankings = Array2::zeros((num_layers, num_neurons));
    let mut ranked_neurons = Array2::zeros((num_layers, num_neurons));

    for ((layer, mut rankings_layer), mut ranked_neurons_layer) in ranking_values
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

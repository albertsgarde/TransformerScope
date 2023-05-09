from typing import List, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional
from transformer_lens import HookedTransformer


def int_to_label(i: int) -> str:
    alpha = "ABCDEFGH"
    return f"{alpha[i//8]}{i%8}"


def calculate_heatmaps(
    model: HookedTransformer,
    blank_probe_normalised: Tensor,
    ownership_probe_normalised: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Shape of both outputs is (num_layers, num_neurons, num_rows, num_cols)"""
    # Output weights for all neurons
    # Shape (num_layers, num_neurons, num_features)
    w_out = model.W_out.detach()
    # Normalize the weights individually for each neuron
    w_out = functional.normalize(w_out, dim=2)

    heatmaps_ownership = (
        w_out[:, :, :, None, None] * ownership_probe_normalised[None, None, :, :, :]
    ).sum(dim=2)
    heatmaps_blank = (
        w_out[:, :, :, None, None] * blank_probe_normalised[None, None, :, :, :]
    ).sum(dim=2)

    return heatmaps_ownership, heatmaps_blank


def calculate_logit_attributions(model: HookedTransformer) -> Tensor:
    # A tensor of shape (num_layers, num_neurons, num_features)
    w_out = model.W_out
    num_layers, num_neurons, num_features = w_out.shape
    # The unembedding matrix without the pass action.
    # Shape (num_features, num_actions-1)
    unembedding_matrix = model.W_U[:, 1:]
    num_features2, num_actions = unembedding_matrix.shape
    assert num_features2 == num_features
    # Set of board position indices affected by actions
    board_positions = list(range(0, 27)) + list(range(29, 35)) + list(range(37, 64))
    assert len(board_positions) == 60
    assert num_actions == 60
    attributions = torch.zeros(num_layers, num_neurons, 64, device=w_out.device)
    attributions[:, :, board_positions] = w_out @ unembedding_matrix
    attributions = attributions.reshape(num_layers, num_neurons, 8, 8)

    assert (
        torch.all(attributions[:, :, 3, 3] == 0)
        and torch.all(attributions[:, :, 3, 4] == 0)
        and torch.all(attributions[:, :, 4, 3] == 0)
        and torch.all(attributions[:, :, 4, 4] == 0)
    )

    return attributions


def neuron_ranking(values: Tensor) -> Tuple[List[List[int]], List[List[int]]]:
    assert values.shape == (8, 2048)

    ranks = []
    sorted_neurons = []
    for layer_values in values:
        neuron_indices = list(enumerate(layer_values))
        neuron_indices.sort(
            reverse=True,
            key=lambda x: x[1],
        )
        sorted_neurons.append([x[0] for x in neuron_indices])

        layer_ranks = np.zeros(len(neuron_indices), dtype=np.int32)
        for neuron_index, (rank, _) in enumerate(neuron_indices):
            layer_ranks[neuron_index] = rank

        assert len([x for x in layer_ranks if x == 0]) == 1
        ranks.append(layer_ranks)

    return ranks, sorted_neurons

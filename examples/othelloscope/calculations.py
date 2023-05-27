from typing import List, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional
from transformer_lens import HookedTransformer
import transformer_scope as ts


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
    raw_attributions = ts.logit_attributions(model)
    num_layers, num_neurons, num_actions = raw_attributions.shape
    board_positions = list(range(0, 27)) + list(range(29, 35)) + list(range(37, 64))
    assert len(board_positions) == 60
    attributions = torch.zeros(
        num_layers, num_neurons, 64, device=raw_attributions.device
    )
    attributions[:, :, board_positions] = raw_attributions[:, :, 1:]
    attributions = attributions.reshape(num_layers, num_neurons, 8, 8)

    assert (
        torch.all(attributions[:, :, 3, 3] == 0)
        and torch.all(attributions[:, :, 3, 4] == 0)
        and torch.all(attributions[:, :, 4, 3] == 0)
        and torch.all(attributions[:, :, 4, 4] == 0)
    )

    return attributions

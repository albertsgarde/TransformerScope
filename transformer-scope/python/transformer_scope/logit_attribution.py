from torch import Tensor
from transformer_lens import HookedTransformer


def mlp_logit_attributions(model: HookedTransformer) -> Tensor:
    """
    Returns a tensor that describes how much each neuron in MLP layers directly affect the logits.

    :param model: A HookedTransformer.
    :type file_loc: HookedTransformer
    :returns: A tensor of shape (num_layers, num_neurons, num_token_values).
    :rtype: Tensor
    """
    # A tensor of shape (num_layers, num_neurons, num_residuals)
    w_out = model.W_out
    num_layers, num_neurons, num_residuals = w_out.shape
    # Shape (num_residuals, num_token_values)
    unembedding_matrix = model.W_U
    num_residuals2, num_token_values = unembedding_matrix.shape
    assert num_residuals2 == num_residuals

    attributions = w_out @ unembedding_matrix
    attr_layers, attr_neurons, attr_token_values = attributions.shape
    assert attr_layers == num_layers
    assert attr_neurons == num_neurons
    assert attr_token_values == num_token_values
    return attributions

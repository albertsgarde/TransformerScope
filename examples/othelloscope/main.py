import os
import shutil

import numpy as np
import torch
import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig

from examples.othelloscope import calculations

from transformer_scope import PayloadBuilder, Payload, Scope

path = os.path.dirname(os.path.realpath(__file__))

torch.set_grad_enabled(False)

USE_CUDA: bool = torch.cuda.is_available()
DEVICE: str = "cuda" if USE_CUDA else "cpu"
print(f"Using device: {DEVICE}")


def to_device(object):
    if USE_CUDA:
        return object.cuda()
    else:
        return object.cpu()


model_config = HookedTransformerConfig(
    n_layers=8,
    d_model=512,
    d_head=64,
    n_heads=8,
    d_mlp=2048,
    d_vocab=61,
    n_ctx=59,
    act_fn="gelu",
    normalization_type="LNPre",
)
# Create the model.
model: HookedTransformer = to_device(HookedTransformer(model_config))
# Download the model parameters.
print("Downloading model parameters...")
model_state = transformer_lens.utils.download_file_from_hf(
    "NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth"
)
print("Loading model parameters...")
# Load the model parameters.
model.load_state_dict(model_state)

print("Loading games...")
# Sequences of actions taken in various games
# Shape: (num_games, length_of_game(60))
board_seqs_int = torch.tensor(
    np.load(os.path.join(path, "board_seqs_int_small.npy")),
    dtype=torch.long,
    device=DEVICE,
)
# We only want to look at the first 50 games and only the first 59 moves of each game.
focus_game_moves = board_seqs_int[:50, :-1]

print("Calculating focus game activations...")
_, focus_cache = model.run_with_cache(focus_game_moves)
# The activations of the neurons in the MLP for each layer during each of the focus games.
focus_game_neuron_activations = torch.stack(
    [focus_cache["post", layer_index, "mlp"][:, :, :] for layer_index in range(8)],
    dim=0,
)
# Shape (num_layers(8), num_neurons(2048), num_games(50), num_moves(59))
focus_game_neuron_activations = focus_game_neuron_activations.transpose(1, 3).transpose(
    2, 3
)

print("Loading full linear probe...")
# Load the linear probe
# This is a probe which predicts the board state from the residual stream.
# The 3 possible outputs are: empty, theirs, ours
# Shape (num_features(size of residual stream=512), num_rows(8), num_cols(8), num_cell_states(3))
linear_probe = torch.load(os.path.join(path, "linear_probe.pt"), map_location=DEVICE)

print("Creating blank and ownership probe...")
# This probe predicts whether is blank or not.
blank_probe = (
    linear_probe[..., 0] - linear_probe[..., 1] * 0.5 - linear_probe[..., 2] * 0.5
)
# This probe predicts whether we own the cell.
ownership_probe = linear_probe[..., 2] - linear_probe[..., 1]

print("Normalising probes...")
# Scale the probes down to be unit norm per feature
blank_probe_normalised = blank_probe / blank_probe.norm(dim=0, keepdim=True)
ownership_probe_normalised = ownership_probe / ownership_probe.norm(dim=0, keepdim=True)

# Set the center blank probes to 0, since they're never blank so the probe is meaningless
blank_probe_normalised[:, [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

print("Calculating heatmaps...")
ownership_heatmap, blank_heatmap = calculations.calculate_heatmaps(
    model, blank_probe_normalised, ownership_probe_normalised
)

print("Calculating attributions...")
attributions = calculations.calculate_logit_attributions(model)

print("Calculating ownership heatmap standard deviations...")
ownership_heatmap_stds = ownership_heatmap.std(dim=(2, 3))

output_path = os.path.join(path, "output")
shutil.rmtree(output_path, ignore_errors=True)
os.makedirs(output_path, exist_ok=True)

print("Creating payload...")
# Load html template for neuron sites.
neuron_template = open(os.path.join(path, "mlp_neuron.html"), "r").read()

# Convert the focus game moves to string labels.
focus_game_moves_str = np.vectorize(calculations.int_to_label)(
    focus_game_moves.detach().cpu().numpy()
).astype("object")

# Create a payload builder with 8 layers and 2048 neurons per layer.
payload_builder = PayloadBuilder(8, 2048)
# Add the template for the neuron sites to the payload builder.
payload_builder.mlp_neuron_template(neuron_template)
# Add values for the ownership heatmap.
payload_builder.add_f32_value(
    "ownership_heatmap", ownership_heatmap.detach().cpu().numpy(), Scope.Neuron
)
# Add values for the blank heatmap.
payload_builder.add_f32_value(
    "blank_heatmap", blank_heatmap.detach().cpu().numpy(), Scope.Neuron
)
# Add values for the attributions.
payload_builder.add_f32_value(
    "logit_attribution_heatmap", attributions.detach().cpu().numpy(), Scope.Neuron
)

# Add the standard deviations of the ownership heatmaps.
payload_builder.add_f32_value(
    "ownership_heatmap_stds",
    ownership_heatmap_stds.detach().cpu().numpy(),
    Scope.Neuron,
)
# Set the standard deviations of the ownership heatmaps as the rank values.
payload_builder.set_rank_values("ownership_heatmap_stds")

# Add activation values in focus games for each neuron.
payload_builder.add_f32_value(
    "focus_game_neuron_activations",
    focus_game_neuron_activations.detach().cpu().numpy(),
    Scope.Neuron,
)
# Add the focus game moves as string values.
payload_builder.add_str_value("focus_game_moves", focus_game_moves_str, Scope.Global)

# Build the payload.
payload = payload_builder.build()

print("Writing payload to file...")
payload.to_file(os.path.join(output_path, "payload"))
print("Payload written to file.")

print("Generating site HTML...")
payload.generate_site_files(output_path)
print("Site HTML generated.")

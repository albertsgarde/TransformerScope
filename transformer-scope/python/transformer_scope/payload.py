import numpy as np

from . import transformer_scope as ts


class Payload:
    def to_file(self, path: str) -> None:
        self.payload.to_file(path)

    def generate_site_files(self, dir_path: str) -> None:
        self.payload.generate_site_files(dir_path)


class PayloadBuilder:
    def __init__(self, num_layers: int, num_mlp_neurons: int):
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_mlp_neurons < 1:
            raise ValueError("num_mlp_neurons must be >= 1")
        self.payload_builder = ts.PayloadBuilder(num_layers, num_mlp_neurons)

    def mlp_neuron_template(self, template: str) -> None:
        self.payload_builder.mlp_neuron_template(template)

    def add_str_value(self, key: str, value: np.ndarray, scope: ts.Scope) -> None:
        self.payload_builder.add_str_value(key, value, scope)

    def add_u32_value(self, key: str, value: np.ndarray, scope: ts.Scope) -> None:
        self.payload_builder.add_u32_value(key, value, scope)

    def add_f32_value(self, key: str, value: np.ndarray, scope: ts.Scope) -> None:
        self.payload_builder.add_f32_value(key, value, scope)

    def set_rank_values(self, key: str) -> None:
        self.payload_builder.set_rank_values(key)

    def build(self) -> Payload:
        payload = Payload()
        payload.payload = self.payload_builder.build()
        return payload

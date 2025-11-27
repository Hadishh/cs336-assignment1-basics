import torch
from src.nn.tranfromer_layer import TransformerLayer


class Transformer(torch.nn.Module):
    def __init__(
        self,
        num_layers,
    ):
        super().__init__()

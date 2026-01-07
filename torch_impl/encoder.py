import torch
import torch.nn as nn
from torch_impl.encoder_block import EncoderBlock

class Encoder(nn.Module):

    def __init__(self, d, numLayers):

        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(d) for _ in range(numLayers)]
        )

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

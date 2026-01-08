import torch.nn as nn
from torch_impl.decoder_block import DecoderBlock


class Decoder(nn.Module):

    def __init__(self, d, numlayers):
        super().__init__()

        self.Layers = nn.ModuleList(
            [DecoderBlock(d) for _ in range(numlayers)]
        )

    def forward(self, X, mask):
        for layer in self.Layers:
            X = layer(X, mask)
        return X
    

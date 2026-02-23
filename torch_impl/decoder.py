import torch.nn as nn
from torch_impl.decoder_block import DecoderBlock
from torch_impl.causal_mask import causal_mask

class Decoder(nn.Module):

    def __init__(self, d, numlayers):
        super().__init__()

        self.Layers = nn.ModuleList(
            [DecoderBlock(d) for _ in range(numlayers)]
        )

    def forward(self, X):
        """
        X: (T, d)
        """
        T = X.size(0)

        # create causal mask once
        mask = causal_mask(T).to(X.device)

        for layer in self.Layers:
            X = layer(X, mask)
        return X
    

from core.decoder_block import Decoderblock

class Decoder:

    def __init__(self, num_layers, d):
        self.layers = [Decoderblock(d) for _ in range(num_layers)]

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

#->Methods don’t have their own self.
#->They all receive the same object when called on that object.
#->self is that object and it is same for all functions 
#of a class.
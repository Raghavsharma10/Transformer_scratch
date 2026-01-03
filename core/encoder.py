from core.encoder_block import Encoderblock


class Encoder:

    def __init__(self, num_layers, d):

        self.layers = [Encoderblock(d) for _ in range(num_layers)] # The list does not contain the objects themselves.it contains addresses pointing to them. 

        ### layers = [
#  ──▶ EncoderBlock object #1
#  ──▶ EncoderBlock object #2
#  ──▶ EncoderBlock object #3
#       ]
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
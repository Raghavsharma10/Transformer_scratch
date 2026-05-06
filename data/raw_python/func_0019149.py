def nmb_weights_hidden(self) -> int:
        """Number of hidden weights.

        >>> from hydpy import ANN
        >>> ann = ANN(None)
        >>> ann(nmb_inputs=2, nmb_neurons=(4, 3, 2), nmb_outputs=3)
        >>> ann.nmb_weights_hidden
        18
        """
        nmb = 0
        for idx_layer in range(self.nmb_layers-1):
            nmb += self.nmb_neurons[idx_layer] * self.nmb_neurons[idx_layer+1]
        return nmb
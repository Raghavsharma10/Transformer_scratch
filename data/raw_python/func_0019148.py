def shape_weights_hidden(self) -> Tuple[int, int, int]:
        """Shape of the array containing the activation of the hidden neurons.

        The first integer value is the number of connection between the
        hidden layers, the second integer value is maximum number of
        neurons of all hidden layers feeding information into another
        hidden layer (all except the last one), and the third integer
        value is the maximum number of the neurons of all hidden layers
        receiving information from another hidden layer (all except the
        first one):

        >>> from hydpy import ANN
        >>> ann = ANN(None)
        >>> ann(nmb_inputs=6, nmb_neurons=(4, 3, 2), nmb_outputs=6)
        >>> ann.shape_weights_hidden
        (2, 4, 3)
        >>> ann(nmb_inputs=6, nmb_neurons=(4,), nmb_outputs=6)
        >>> ann.shape_weights_hidden
        (0, 0, 0)
        """
        if self.nmb_layers > 1:
            nmb_neurons = self.nmb_neurons
            return (self.nmb_layers-1,
                    max(nmb_neurons[:-1]),
                    max(nmb_neurons[1:]))
        return 0, 0, 0
def backprop(self, input_data, df_output, cache=None):
        """ Backpropagate through the hidden layer

        **Parameters:**

        input_data : ``GPUArray``
            Inpute data to perform dropout on.

        df_output : ``GPUArray``
            Gradients with respect to the output of this layer
            (received from the layer above).

        cache : list of ``GPUArray``
            Cache obtained from forward pass. If the cache is
            provided, then the activations are not recalculated.

        **Returns:**

        gradients : empty tuple
            Gradients are empty since this layer has no parameters.

        df_input : ``GPUArray``
            Gradients with respect to the input.
        """

        if self.compute_input_gradients:            
            apply_dropout_mask(df_output, dropout_mask)

        return tuple(), df_output
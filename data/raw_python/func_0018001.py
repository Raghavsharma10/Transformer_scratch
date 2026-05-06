def backprop(self, input_data, df_output, cache=None):
        """ Backpropagate through the hidden layer

        **Parameters:**

        input_data : ``GPUArray``
            Input data to compute activations for.

        df_output : ``GPUArray``
            Gradients with respect to the activations of this layer
            (received from the layer above).

        cache : list of ``GPUArray``
            Cache obtained from forward pass. If the cache is
            provided, then the activations are not recalculated.

        **Returns:**

        gradients : tuple of ``GPUArray``
            Gradients with respect to the weights and biases in the
            form ``(df_weights, df_biases)``.

        df_input : ``GPUArray``
            Gradients with respect to the input.
        """

        # Get cache if it wasn't provided
        if cache is None:
            cache = self.feed_forward(input_data,
                                      prediction=False)

        if len(cache) == 2:
            activations, dropout_mask = cache
        else:
            activations = cache[0]

        # Multiply the binary mask with the incoming gradients
        if self.dropout > 0 and dropout_mask is not None:
            apply_dropout_mask(df_output, dropout_mask)

        # Get gradient wrt activation function
        df_activations = self.df(activations)
        delta = mult_matrix(df_activations, df_output)

        # Gradient wrt weights
        df_W = linalg.dot(input_data, delta, transa='T')
        # Gradient wrt bias
        df_b = matrix_sum_out_axis(delta, 0)
        # Gradient wrt inputs
        df_input = linalg.dot(delta, self.W, transb='T')

        # L1 weight decay
        if self.l1_penalty_weight:
            df_W += self.l1_penalty_weight * sign(self.W)

        # L2 weight decay
        if self.l2_penalty_weight:
            df_W += self.l2_penalty_weight * self.W

        return (df_W, df_b), df_input
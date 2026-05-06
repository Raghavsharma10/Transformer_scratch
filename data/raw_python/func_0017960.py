def backprop(self, input_data, targets,
                 cache=None):
        """ Backpropagate through the logistic layer.

        **Parameters:**

        input_data : ``GPUArray``
            Inpute data to compute activations for.

        targets : ``GPUArray``
            The target values of the units.

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

        if cache is not None:
            activations = cache
        else:
            activations = self.feed_forward(input_data, prediction=False)

        if activations.shape != targets.shape:
            raise ValueError('Activations (shape = %s) and targets (shape = %s) are different sizes' %
                             (activations.shape, targets.shape))

        delta = substract_matrix(activations, targets)
        nan_to_zeros(delta, delta)

        # Gradient wrt weights
        df_W = linalg.dot(input_data, delta, transa='T')
        # Gradient wrt bias
        df_b = matrix_sum_out_axis(delta, 0)

        # Gradient wrt input
        df_input = linalg.dot(delta, self.W, transb='T')

        # L1 penalty
        if self.l1_penalty_weight:
            df_W += self.l1_penalty_weight * sign(self.W)

        # L2 penalty
        if self.l2_penalty_weight:
            df_W += self.l2_penalty_weight * self.W

        return (df_W, df_b), df_input
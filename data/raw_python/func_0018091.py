def feed_forward(self, input_data, prediction=False):
        """Propagate forward through the layer.

        **Parameters:**

        input_data : ``GPUArray``
            Inpute data to compute activations for.

        prediction : bool, optional
            Whether to use prediction model. Only relevant when using
            dropout. If true, then weights are multiplied by
            1 - dropout if the layer uses dropout.

        **Returns:**
        
        activations : ``GPUArray``
            The activations of the output units.
        """

        if input_data.shape[1] != self.W.shape[0]:
            raise ValueError('Number of outputs from previous layer (%d) '
                             'does not match number of inputs to this layer (%d)' %
                             (input_data.shape[1], self.W.shape[0]))

        activations = linalg.dot(input_data, self.W)
        activations = add_vec_to_mat(activations, self.b, inplace=True)

        return activations
def feed_forward(self, input_data, prediction=False):
        """Propagate forward through the layer

        **Parameters:**

        input_data : ``GPUArray``
            Inpute data to perform dropout on.

        prediction : bool, optional
            Whether to use prediction model. If true, then the data is
            scaled by ``1 - dropout_probability`` uses dropout.

        **Returns:**
        
        dropout_data : ``GPUArray``
            The data after performing dropout.
        """

        if input_data.shape[1] != self.n_in:
            raise ValueError('Number of outputs from previous layer (%d) '
                             'does not match number of inputs to this layer (%d)' %
                             (input_data.shape[1], self.n_in))

        if not prediction:
            dropout_input = gpuarray.empty_like(input_data)
            dropout_mask = sample_dropout_mask(input_data,
                                               self.dropout_probability, target=dropout_input
                                           )
            return dropout_input, dropout_mask
        else:
            return (input_data * (1 - self.dropout_probability),)
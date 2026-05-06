def parameters(self, value):
        """ Used to set all of the model's parameters to new values.

        **Parameters:**

        value : array_like
            New values for the model parameters. Must be of length
            ``self.n_parameters``.
        """
    
        if len(value) != self.n_parameters:
            raise ValueError("Incorrect length of parameter vector. "
                             "Model has %d parameters, but got %d" %
                             (self.n_parameters, len(value)))

        i = 0
        for hl in self.hidden_layers:
            hl.parameters = value[i:i + hl.n_parameters]
            i += hl.n_parameters

        self.top_layer.parameters = value[-self.top_layer.n_parameters:]
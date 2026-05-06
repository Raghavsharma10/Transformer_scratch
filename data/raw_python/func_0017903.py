def training_pass(self, input_data, targets):
        """ Perform a full forward and backward pass through the model.

        **Parameters:**

        input_data : GPUArray
            Data to train the model with.

        targets : GPUArray
            Training targets.

        **Returns:**

        loss : float
            Value of loss function as evaluated on the data and targets.

        gradients : list of GPUArray
            Gradients obtained from backpropagation in the backward pass.
        """

        # Forward pass
        loss, hidden_cache, logistic_cache = self.evaluate(
            input_data, targets, return_cache=True, prediction=False)

        if not np.isfinite(loss):
            raise ValueError('Infinite activations!')

        # Backpropagation
        if self.hidden_layers:
            hidden_activations = hidden_cache[-1][0]
        else:
            hidden_activations = input_data

        df_top_layer = \
          self.top_layer.backprop(hidden_activations, targets,
                                  cache=logistic_cache)
        gradients = list(df_top_layer[0][::-1])
        df_hidden = df_top_layer[1]

        if self.hidden_layers:
            hidden_inputs = [input_data] + [c[0] for c in hidden_cache[:-1]]            
            for hl, hc, hi in \
                zip(self.hidden_layers[::-1], hidden_cache[::-1],
                    hidden_inputs[::-1]):
                g, df_hidden = hl.backprop(hi, df_hidden, cache=hc)
                gradients.extend(g[::-1])

        gradients.reverse()

        return loss, gradients
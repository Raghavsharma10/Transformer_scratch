def feed_forward(self, input_data, return_cache=False, prediction=True):
        """ Run data forward through the model.

        **Parameters:**

        input_data : GPUArray
            Data to run through the model.

        return_cache : bool, optional
            Whether to return the intermediary results.

        prediction : bool, optional
            Whether to run in prediction mode. Only relevant when
            using dropout. If true, weights are multiplied by 1 - dropout.
            If false, then half of hidden units are randomly dropped and
            the dropout mask is returned in case ``return_cache==True``.

        **Returns:**
        
        prediction : GPUArray
            Predictions from the model.

        cache : list of GPUArray, only returned if ``return_cache == True``
            Results of intermediary computations.    
        """

        hidden_cache = None     # Create variable in case there are no hidden layers
        if self.hidden_layers:
            # Forward pass
            hidden_cache = []
            for i in range(len(self.hidden_layers)):
                hidden_activations = hidden_cache[i - 1][0] if i else input_data
                # Use dropout predict if previous layer has dropout
                hidden_cache.append(self.hidden_layers[i]
                                    .feed_forward(hidden_activations,
                                                  prediction=prediction))

            hidden_activations = hidden_cache[-1][0]

        else:
            hidden_activations = input_data

        # Use dropout_predict if last hidden layer has dropout
        activations = \
          self.top_layer.feed_forward(hidden_activations,
                                      prediction=False)

        if return_cache:
            return activations, hidden_cache
        return activations
def evaluate(self, input_data, targets,
                 return_cache=False, prediction=True):
        """ Evaluate the loss function without computing gradients.

        **Parameters:**

        input_data : GPUArray
            Data to evaluate

        targets: GPUArray
            Targets

        return_cache : bool, optional
            Whether to return intermediary variables from the
            computation and the hidden activations.

        prediction : bool, optional
            Whether to use prediction model. Only relevant when using
            dropout. If true, then weights are multiplied by
            1 - dropout if the layer uses dropout.

        **Returns:**

        loss : float
            The value of the loss function.

        hidden_cache : list, only returned if ``return_cache == True``
            Cache as returned by :meth:`hebel.models.NeuralNet.feed_forward`.

        activations : list, only returned if ``return_cache == True``
            Hidden activations as returned by
            :meth:`hebel.models.NeuralNet.feed_forward`.
        """

        # Forward pass
        activations, hidden_cache = self.feed_forward(
            input_data, return_cache=True, prediction=prediction)

        loss = self.top_layer.train_error(None,
            targets, average=False, cache=activations,
            prediction=prediction)

        for hl in self.hidden_layers:
            if hl.l1_penalty_weight: loss += hl.l1_penalty
            if hl.l2_penalty_weight: loss += hl.l2_penalty

        if self.top_layer.l1_penalty_weight: loss += self.top_layer.l1_penalty
        if self.top_layer.l2_penalty_weight: loss += self.top_layer.l2_penalty

        if not return_cache:
            return loss
        else:
            return loss, hidden_cache, activations
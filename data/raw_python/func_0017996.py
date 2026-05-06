def backprop(self, input_data, targets, cache=None):
        """Compute gradients for each task and combine the results.

        **Parameters:**

        input_data : ``GPUArray``
            Inpute data to compute activations for.

        targets : ``GPUArray``
            The target values of the units.

        cache : list of ``GPUArray``
            Cache obtained from forward pass. If the cache is
            provided, then the activations are not recalculated.

        **Returns:**

        gradients : list
            Gradients with respect to the weights and biases for each task

        df_input : ``GPUArray``
            Gradients with respect to the input, obtained by adding
            the gradients with respect to the inputs from each task,
            weighted by ``MultitaskTopLayer.task_weights``.
        """

        df_input = gpuarray.zeros_like(input_data)

        if cache is None: cache = self.n_tasks * [None]

        gradients = []
        for targets_task, cache_task, task, task_weight  in \
          izip(targets, cache, self.tasks, self.task_weights):
            gradients_task, df_input_task = \
              task.backprop(input_data, targets_task,
                            cache_task)

            df_input = df_input.mul_add(1., df_input_task, task_weight)

            gradients.extend(gradients_task)

        return gradients, df_input
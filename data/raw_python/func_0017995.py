def feed_forward(self, input_data, prediction=False):
        """Call ``feed_forward`` for each task and combine the activations.

        Passes ``input_data`` to all tasks and returns the activations
        as a list.
    
        **Parameters:**

        input_data : ``GPUArray``
            Inpute data to compute activations for.

        prediction : bool, optional
            Whether to use prediction model. Only relevant when using
            dropout. If true, then weights are multiplied by
            1 - dropout if the layer uses dropout.

        **Returns:**
        
        activations : list of ``GPUArray``
            The activations of the output units, one element for each task.
        """

        activations = []

        for task in self.tasks:
            activations_task = task.feed_forward(input_data, prediction)
            activations.append(activations_task)

        return activations
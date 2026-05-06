def parameters(self, value):
        """Update the parameters.

        ``value`` must be a list/tuple of length
        ``MultitaskTopLayer.n_tasks``, each element of which must have
        the correct number of parameters for the task.
        """

        assert len(value) == self.n_parameters
        i = 0
        for task in self.tasks:
            task.parameters = value[i:i + task.n_parameters]
            i += task.n_parameters
def parameters(self):
        """Return a list where each element contains the parameters for a task.
        """
        parameters = []
        for task in self.tasks:
            parameters.extend(task.parameters)
        return parameters
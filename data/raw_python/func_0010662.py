def eval(self, data, data_store, *, exclude=None):
        """ Return a new object in which callable parameters have been evaluated.

        Native types are not touched and simply returned, while callable methods are
        executed and their return value is returned.

        Args:
            data (MultiTaskData): The data object that has been passed from the
                                  predecessor task.
            data_store (DataStore): The persistent data store object that allows the task
                                    to store data for access across the current workflow
                                    run.
            exclude (list): List of key names as strings that should be excluded from
                            the evaluation.

        Returns:
            TaskParameters: A new TaskParameters object with the callable parameters
                            replaced by their return value.
        """
        exclude = [] if exclude is None else exclude

        result = {}
        for key, value in self.items():
            if key in exclude:
                continue

            if value is not None and callable(value):
                result[key] = value(data, data_store)
            else:
                result[key] = value
        return TaskParameters(result)
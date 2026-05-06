def eval_single(self, key, data, data_store):
        """ Evaluate the value of a single parameter taking into account callables .

        Native types are not touched and simply returned, while callable methods are
        executed and their return value is returned.

        Args:
            key (str): The name of the parameter that should be evaluated.
            data (MultiTaskData): The data object that has been passed from the
                                  predecessor task.
            data_store (DataStore): The persistent data store object that allows the task
                                    to store data for access across the current workflow
                                    run.

        """
        if key in self:
            value = self[key]
            if value is not None and callable(value):
                return value(data, data_store)
            else:
                return value
        else:
            raise AttributeError()
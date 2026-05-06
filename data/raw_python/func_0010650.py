def add_dataset(self, task_name, dataset=None, *, aliases=None):
        """ Add a new dataset to the MultiTaskData.

        Args:
            task_name (str): The name of the task from which the dataset was received.
            dataset (TaskData): The dataset that should be added.
            aliases (list): A list of aliases that should be registered with the dataset.
        """
        self._datasets.append(dataset if dataset is not None else TaskData())
        last_index = len(self._datasets) - 1
        self._aliases[task_name] = last_index

        if aliases is not None:
            for alias in aliases:
                self._aliases[alias] = last_index

        if len(self._datasets) == 1:
            self._default_index = 0
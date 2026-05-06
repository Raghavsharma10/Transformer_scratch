def run(self, data, store, signal, context, **kwargs):
        """ The main run method of the Python task.

        Args:
            data (:class:`.MultiTaskData`): The data object that has been passed from the
                predecessor task.
            store (:class:`.DataStoreDocument`): The persistent data store object that allows the
                task to store data for access across the current workflow run.
            signal (TaskSignal): The signal object for tasks. It wraps the construction
                and sending of signals into easy to use methods.
            context (TaskContext): The context in which the tasks runs.

        Returns:
            Action: An Action object containing the data that should be passed on
                to the next task and optionally a list of successor tasks that
                should be executed.
        """
        if self._callback is not None:
            result = self._callback(data, store, signal, context, **kwargs)
            return result if result is not None else Action(data)
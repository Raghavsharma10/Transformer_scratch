def register(self, func):
        """
        Register a task. Typically used as a decorator to the task function.

        If a task by that name already exists,
        a TaskAlreadyRegistered exception is raised.
        :param func: func to register as an ape task
        :return: invalid accessor
        """

        if hasattr(self._tasks, func.__name__):
            raise TaskAlreadyRegistered(func.__name__)
        setattr(self._tasks, func.__name__, func)
        return _get_invalid_accessor(func.__name__)
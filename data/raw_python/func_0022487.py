def _validate_entities(self, tasks):
        """
        Purpose: Validate whether the 'tasks' is of type set. Validate the description of each Task.
        """

        if not tasks:
            raise TypeError(expected_type=Task, actual_type=type(tasks))

        if not isinstance(tasks, set):

            if not isinstance(tasks, list):
                tasks = set([tasks])
            else:
                tasks = set(tasks)

        for t in tasks:

            if not isinstance(t, Task):
                raise TypeError(expected_type=Task, actual_type=type(t))

        return tasks
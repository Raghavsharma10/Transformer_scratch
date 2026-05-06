def add_tasks(self, value):
        """
        Adds tasks to the existing set of tasks of the Stage

        :argument: set of tasks
        """
        tasks = self._validate_entities(value)
        self._tasks.update(tasks)
        self._task_count = len(self._tasks)
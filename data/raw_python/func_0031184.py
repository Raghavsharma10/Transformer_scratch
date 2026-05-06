def add_task(self, task):
        """
        Add a task to the scheduler.

        task: The task to add.
        """
        if not self._valid_name(task.name):
            raise ValueError(task.name)

        self._tasks[task.name] = task

        incomplete_dependencies = set()

        for dependency in task.dependencies:
            if not self._valid_name(dependency) or dependency in self._failed:
                # there may already be tasks dependent on this one.
                self._cascade_failure(task.name)

                break

            if dependency not in self._completed:
                incomplete_dependencies.add(dependency)
        else:  # task hasn't failed
            try:
                self._graph.add(task.name, incomplete_dependencies)
            except ValueError:
                self._cascade_failure(task.name)
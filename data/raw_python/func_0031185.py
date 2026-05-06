def start_task(self, name=None):
        """
        Start a task.

        Returns the task that was started (or None if no task has been
            started).

        name: (optional, None) The task to start. If a name is given,
            Scheduler will attempt to start the task (and raise an
            exception if the task doesn't exist or isn't runnable). If
            no name is given, a task will be chosen arbitrarily
        """
        if name is None:
            for possibility in self._graph.roots:
                if possibility not in self._running:
                    name = possibility
                    break
            else:  # all tasks blocked/running/completed/failed
                return None
        else:
            if name not in self._graph.roots or name in self._running:
                raise ValueError(name)

        self._running.add(name)

        return self._tasks[name]
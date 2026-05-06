def end_task(self, name, success=True):
        """
        End a running task. Raises an exception if the task isn't
        running.

        name: The name of the task to complete.
        success: (optional, True) Whether the task was successful.
        """
        self._running.remove(name)

        if success:
            self._completed.add(name)
            self._graph.remove(name, strategy=Strategy.orphan)
        else:
            self._cascade_failure(name)
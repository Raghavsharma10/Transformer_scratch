def _cascade_failure(self, name):
        """
        Mark a task (and anything that depends on it) as failed.

        name: The name of the offending task
        """
        if name in self._graph:
            self._failed.update(
                self._graph.remove(name, strategy=Strategy.remove)
            )
        else:
            self._failed.add(name)
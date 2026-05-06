def fail_remaining(self):
        """
        Mark all unfinished tasks (including currently running ones) as
        failed.
        """
        self._failed.update(self._graph.nodes)
        self._graph = Graph()
        self._running = set()
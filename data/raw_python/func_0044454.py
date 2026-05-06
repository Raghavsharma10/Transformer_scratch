def of_think(self, think):
        """
        Simulate the worker processing the task for the specified amount of time.
        The worker is not released and the task is not paused.
        """
        return self._compute(
            duration=think.duration,
            after=self.continuation)
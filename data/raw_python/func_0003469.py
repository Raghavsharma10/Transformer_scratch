def register(self, result):
        """
        Register an EventualResult.

        May be called in any thread.
        """
        if self._stopped:
            raise ReactorStopped()
        self._results.add(result)
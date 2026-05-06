def store(self, deferred_result):
        """
        Store a EventualResult.

        Return an integer, a unique identifier that can be used to retrieve
        the object.
        """
        self._counter += 1
        self._stored[self._counter] = deferred_result
        return self._counter
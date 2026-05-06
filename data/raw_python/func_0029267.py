def add_result(self, result):
        """Add an already-constructed :class:`~.TestResult` to this
        :class:`~.ResultCollector`.

        This may be used when collecting results created by other
        ResultCollectors (e.g. in subprocesses).

        """
        for handler in self._handlers:
            handler(result)
        if self._successful and result.status not in _successful_results:
            self._successful = False
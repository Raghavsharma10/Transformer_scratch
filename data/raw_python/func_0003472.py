def _set_result(self, result):
        """
        Set the result of the EventualResult, if not already set.

        This can only happen in the reactor thread, either as a result of
        Deferred firing, or as a result of ResultRegistry.stop(). So, no need
        for thread-safety.
        """
        if self._result_set.isSet():
            return
        self._value = result
        self._result_set.set()
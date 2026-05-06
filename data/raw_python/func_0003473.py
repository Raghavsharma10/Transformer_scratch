def _result(self, timeout=None):
        """
        Return the result, if available.

        It may take an unknown amount of time to return the result, so a
        timeout option is provided. If the given number of seconds pass with
        no result, a TimeoutError will be thrown.

        If a previous call timed out, additional calls to this function will
        still wait for a result and return it if available. If a result was
        returned on one call, additional calls will return/raise the same
        result.
        """
        if timeout is None:
            warnings.warn(
                "Unlimited timeouts are deprecated.",
                DeprecationWarning,
                stacklevel=3)
            # Queue.get(None) won't get interrupted by Ctrl-C...
            timeout = 2**31
        self._result_set.wait(timeout)
        # In Python 2.6 we can't rely on the return result of wait(), so we
        # have to check manually:
        if not self._result_set.is_set():
            raise TimeoutError()
        self._result_retrieved = True
        return self._value
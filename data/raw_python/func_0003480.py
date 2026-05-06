def run_in_reactor(self, function):
        """
        A decorator that ensures the wrapped function runs in the
        reactor thread.

        When the wrapped function is called, an EventualResult is returned.
        """
        result = self._run_in_reactor(function)
        # Backwards compatibility; use __wrapped__ instead.
        try:
            result.wrapped_function = function
        except AttributeError:
            pass
        return result
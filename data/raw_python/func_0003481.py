def wait_for_reactor(self, function):
        """
        DEPRECATED, use wait_for(timeout) instead.

        A decorator that ensures the wrapped function runs in the reactor
        thread.

        When the wrapped function is called, its result is returned or its
        exception raised. Deferreds are handled transparently.
        """
        warnings.warn(
            "@wait_for_reactor is deprecated, use @wait_for instead",
            DeprecationWarning,
            stacklevel=2)
        # This will timeout, in theory. In practice the process will be dead
        # long before that.
        return self.wait_for(2**31)(function)
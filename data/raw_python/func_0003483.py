def in_reactor(self, function):
        """
        DEPRECATED, use run_in_reactor.

        A decorator that ensures the wrapped function runs in the reactor
        thread.

        The wrapped function will get the reactor passed in as a first
        argument, in addition to any arguments it is called with.

        When the wrapped function is called, an EventualResult is returned.
        """
        warnings.warn(
            "@in_reactor is deprecated, use @run_in_reactor",
            DeprecationWarning,
            stacklevel=2)

        @self.run_in_reactor
        @wraps(function)
        def add_reactor(*args, **kwargs):
            return function(self._reactor, *args, **kwargs)

        return add_reactor
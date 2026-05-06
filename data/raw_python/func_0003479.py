def _run_in_reactor(self, function, _, args, kwargs):
        """
        Implementation: A decorator that ensures the wrapped function runs in
        the reactor thread.

        When the wrapped function is called, an EventualResult is returned.
        """

        def runs_in_reactor(result, args, kwargs):
            d = maybeDeferred(function, *args, **kwargs)
            result._connect_deferred(d)

        result = EventualResult(None, self._reactor)
        self._registry.register(result)
        self._reactor.callFromThread(runs_in_reactor, result, args, kwargs)
        return result
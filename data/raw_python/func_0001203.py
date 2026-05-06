def connect(self, cback, subscribers=None, instance=None):
        """Add  a function or a method as an handler of this signal.
        Any handler added can be a coroutine.

        :param cback: the callback (or *handler*) to be added to the set
        :returns: ``None`` or the value returned by the corresponding wrapper
        """
        if subscribers is None:
            subscribers = self.subscribers
        # wrapper
        if self._fconnect is not None:
            def _connect(cback):
                self._connect(subscribers, cback)

            notify = partial(self._notify_one, instance)
            if instance is not None:
                result = self._fconnect(instance, cback, subscribers,
                                        _connect, notify)
            else:
                result = self._fconnect(cback, subscribers, _connect, notify)
            if inspect.isawaitable(result):
                result = pull_result(result)
        else:
            self._connect(subscribers, cback)
            result = None
        return result
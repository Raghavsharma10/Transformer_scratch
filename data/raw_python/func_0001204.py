def disconnect(self, cback, subscribers=None, instance=None):
        """Remove a previously added function or method from the set of the
        signal's handlers.

        :param cback: the callback (or *handler*) to be added to the set
        :returns: ``None`` or the value returned by the corresponding wrapper
        """
        if subscribers is None:
            subscribers = self.subscribers
        # wrapper
        if self._fdisconnect is not None:
            def _disconnect(cback):
                self._disconnect(subscribers, cback)

            notify = partial(self._notify_one, instance)
            if instance is not None:
                result = self._fdisconnect(instance, cback, subscribers,
                                           _disconnect, notify)
            else:
                result = self._fdisconnect(cback, subscribers, _disconnect,
                                           notify)
            if inspect.isawaitable(result):
                result = pull_result(result)
        else:
            self._disconnect(subscribers, cback)
            result = None
        return result
def watchSignal(self, signal):
        """
        Setup provisions to watch a specified signal

        :param signal:
            The :class:`Signal` to watch for.

        After calling this method you can use :meth:`assertSignalFired()`
        and :meth:`assertSignalNotFired()` with the same signal.
        """
        self._extend_state()

        def signal_handler(*args, **kwargs):
            self._events_seen.append((signal, args, kwargs))
        signal.connect(signal_handler)
        if hasattr(self, 'addCleanup'):
            self.addCleanup(signal.disconnect, signal_handler)
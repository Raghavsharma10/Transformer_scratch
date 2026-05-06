def block(self, callback=None):
        """Block this emitter. Any attempts to emit an event while blocked
        will be silently ignored. If *callback* is given, then the emitter
        is only blocked for that specific callback.

        Calls to block are cumulative; the emitter must be unblocked the same
        number of times as it is blocked.
        """
        self._blocked[callback] = self._blocked.get(callback, 0) + 1
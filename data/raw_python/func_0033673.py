def run(self):
        """
        Process all callbacks, until `stop()` is called. Intended to run in
        its own thread.
        """
        while True:
            msg = self._queue.get()
            if msg is _SHUTDOWN:
                break
            event, args, kwargs = msg
            self._logger('<< %s', event)
            for func in self._callbacks.get(event, []):
                func(*args, **kwargs)
def register(self, event, fn):
        """
        Tell the object to run `fn` whenever a message of type `event` is
        received.
        """
        self._callbacks.setdefault(event, []).append(fn)
        return fn
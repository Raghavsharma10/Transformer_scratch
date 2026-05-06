def disconnect(self, callback=None):
        """Disconnect a callback from this emitter.

        If no callback is specified, then *all* callbacks are removed.
        If the callback was not already connected, then the call does nothing.
        """
        if callback is None:
            self._callbacks = []
            self._callback_refs = []
        else:
            callback = self._normalize_cb(callback)
            if callback in self._callbacks:
                idx = self._callbacks.index(callback)
                self._callbacks.pop(idx)
                self._callback_refs.pop(idx)
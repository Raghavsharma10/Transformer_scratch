def put(self, event, *args, **kwargs):
        """
        Schedule a callback for `event`, passing `args` and `kwargs` to each
        registered callback handler.
        """
        self._queue.put((event, args, kwargs))
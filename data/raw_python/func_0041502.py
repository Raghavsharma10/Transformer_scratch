def listeners_iter(self):
        """Return an iterator over the mapping of event => listeners bound.

        The listener list(s) returned should **not** be mutated.

        NOTE(harlowja): Each listener in the yielded (event, listeners)
        tuple is an instance of the :py:class:`~.Listener`  type, which
        itself wraps a provided callback (and its details filter
        callback, if any).
        """
        topics = set(six.iterkeys(self._topics))
        while topics:
            event_type = topics.pop()
            try:
                yield event_type, self._topics[event_type]
            except KeyError:
                pass
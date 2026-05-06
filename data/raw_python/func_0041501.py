def copy(self):
        """Clones this notifier (and its bound listeners)."""
        c = copy.copy(self)
        c._topics = {}
        c._lock = threading.Lock()
        topics = set(six.iterkeys(self._topics))
        while topics:
            event_type = topics.pop()
            try:
                listeners = self._topics[event_type]
                c._topics[event_type] = list(listeners)
            except KeyError:
                pass
        return c
def deregister(self, event_type, callback, details_filter=None):
        """Remove a single listener bound to event ``event_type``.

        :param event_type: deregister listener bound to event_type
        :param callback: callback that was used during registration
        :param details_filter: details filter that was used during
                               registration

        :returns: if a listener was deregistered
        :rtype: boolean
        """
        with self._lock:
            listeners = self._topics.get(event_type, [])
            for i, listener in enumerate(listeners):
                if listener.is_equivalent(callback,
                                          details_filter=details_filter):
                    listeners.pop(i)
                    return True
            return False
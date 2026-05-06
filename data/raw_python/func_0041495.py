def is_registered(self, event_type, callback, details_filter=None):
        """Check if a callback is registered.

        :param event_type: event type callback was registered to
        :param callback: callback that was used during registration
        :param details_filter: details filter that was used during
                               registration

        :returns: if the callback is registered
        :rtype: boolean
        """
        listeners = self._topics.get(event_type, [])
        for listener in listeners:
            if listener.is_equivalent(callback, details_filter=details_filter):
                return True
        return False
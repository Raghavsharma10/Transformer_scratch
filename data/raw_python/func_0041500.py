def deregister_by_uuid(self, event_type, uuid):
        """Remove a single listener bound to event ``event_type``.

        :param event_type: deregister listener bound to event_type
        :param uuid: uuid of listener to remove

        :returns: if the listener was deregistered
        :rtype: boolean
        """
        with self._lock:
            listeners = self._topics.get(event_type, [])
            for i, listener in enumerate(listeners):
                if listener.uuid == uuid:
                    listeners.pop(i)
                    return True
            return False
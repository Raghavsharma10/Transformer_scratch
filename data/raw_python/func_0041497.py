def notify(self, event_type, details):
        """Notify about an event occurrence.

        All callbacks registered to receive notifications about given
        event type will be called. If the provided event type can not be
        used to emit notifications (this is checked via
        the :meth:`.can_be_registered` method) then a value error will be
        raised.

        :param event_type: event type that occurred
        :param details: additional event details *dictionary* passed to
                        callback keyword argument with the same name
        :type details: dictionary

        :returns: a future object that will have a result named tuple with
                  contents being (total listeners called, how many listeners
                  were **successfully** called, how many listeners
                  were not **successfully** called); do note that the result
                  may be delayed depending on internal executor used.
        """
        if not self.can_trigger_notification(event_type):
            raise ValueError("Event type '%s' is not allowed to trigger"
                             " notifications" % event_type)
        listeners = list(self._topics.get(self.ANY, []))
        listeners.extend(self._topics.get(event_type, []))
        if not details:
            details = {}
        fut = self._executor.submit(self._do_dispatch, listeners,
                                    event_type, details)
        return fut
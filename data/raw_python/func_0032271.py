def emit_event(self, event_name, event_body):
        """
        Publishes an event of type ``event_name`` to all subscribers, having the body
        ``event_body``. The event is pushed through all available event transports.

        The event body must be a Python object that can be represented as a JSON.

        :param event_name: a ``str`` representing the event type
        :param event_body: a Python object that can be represented as JSON.

        .. versionadded:: 0.5.0

        .. versionchanged:: 0.10.0
            Added parameter broadcast
        """

        for transport in self.event_transports:
            transport.emit_event(event_name, event_body)
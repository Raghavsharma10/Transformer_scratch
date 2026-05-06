def register(self, event_type: str, handler: Any, **detail: Any) -> None:
        """
        Register a new handler for a specific :class:`slack.events.Event` `type` (See `slack event types documentation
        <https://api.slack.com/events>`_ for a list of event types).

        The arbitrary keyword argument is used as a key/value pair to compare against what is in the incoming
        :class:`slack.events.Event`

        Args:
            event_type: Event type the handler is interested in
            handler: Callback
            **detail: Additional key for routing
        """
        LOG.info("Registering %s, %s to %s", event_type, detail, handler)
        if len(detail) > 1:
            raise ValueError("Only one detail can be provided for additional routing")
        elif not detail:
            detail_key, detail_value = "*", "*"
        else:
            detail_key, detail_value = detail.popitem()

        if detail_key not in self._routes[event_type]:
            self._routes[event_type][detail_key] = {}

        if detail_value not in self._routes[event_type][detail_key]:
            self._routes[event_type][detail_key][detail_value] = []

        self._routes[event_type][detail_key][detail_value].append(handler)
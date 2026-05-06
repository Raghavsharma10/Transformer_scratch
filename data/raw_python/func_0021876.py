def dispatch(self, event: Event) -> Iterator[Any]:
        """
        Yields handlers matching the routing of the incoming :class:`slack.events.Event`.

        Args:
            event: :class:`slack.events.Event`

        Yields:
            handler
        """
        LOG.debug('Dispatching event "%s"', event.get("type"))
        if event["type"] in self._routes:
            for detail_key, detail_values in self._routes.get(
                event["type"], {}
            ).items():
                event_value = event.get(detail_key, "*")
                yield from detail_values.get(event_value, [])
        else:
            return
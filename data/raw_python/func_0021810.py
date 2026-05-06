def dispatch(self, command: Command) -> Iterator[Any]:
        """
        Yields handlers matching the incoming :class:`slack.actions.Command`.

        Args:
            command: :class:`slack.actions.Command`

        Yields:
            handler
        """
        LOG.debug("Dispatching command %s", command["command"])
        for callback in self._routes[command["command"]]:
            yield callback
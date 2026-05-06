def register(
        self,
        pattern: str,
        handler: Any,
        flags: int = 0,
        channel: str = "*",
        subtype: Optional[str] = None,
    ) -> None:
        """
        Register a new handler for a specific :class:`slack.events.Message`.

        The routing is based on regex pattern matching the message text and the incoming slack channel.

        Args:
            pattern: Regex pattern matching the message text.
            handler: Callback
            flags: Regex flags.
            channel: Slack channel ID. Use * for any.
            subtype: Message subtype
        """
        LOG.debug('Registering message endpoint "%s: %s"', pattern, handler)
        match = re.compile(pattern, flags)

        if subtype not in self._routes[channel]:
            self._routes[channel][subtype] = dict()

        if match in self._routes[channel][subtype]:
            self._routes[channel][subtype][match].append(handler)
        else:
            self._routes[channel][subtype][match] = [handler]
def from_rtm(cls, raw_event: MutableMapping) -> "Event":
        """
        Create an event with data coming from the RTM API.

        If the event type is a message a :class:`slack.events.Message` is returned.

        Args:
            raw_event: JSON decoded data from the RTM API

        Returns:
            :class:`slack.events.Event` or :class:`slack.events.Message`
        """
        if raw_event["type"].startswith("message"):
            return Message(raw_event)
        else:
            return Event(raw_event)
def from_event(cls, event):
        """Create a message from an event

        :param event: the event that was received of type ``pubmsg`` or ``privmsg``
        :type event: :class:`Event3`
        :returns: a message that resembles the event
        :rtype: :class:`Message3`
        :raises: None
        """
        source = Chatter(event.source)
        return cls(source, event.target, event.arguments[0], event.tags)
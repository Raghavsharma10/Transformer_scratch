def _handle_message(self, tags, source, command, target, msg):
        """Construct the correct events and handle them

        :param tags: the tags of the message
        :type tags: :class:`list` of :class:`message.Tag`
        :param source: the sender of the message
        :type source: :class:`str`
        :param command: the event type
        :type command: :class:`str`
        :param target: the target of the message
        :type target: :class:`str`
        :param msg: the content
        :type msg: :class:`str`
        :returns: None
        :rtype: None
        :raises: None
        """
        if isinstance(msg, tuple):
            if command in ["privmsg", "pubmsg"]:
                command = "ctcp"
            else:
                command = "ctcpreply"

            msg = list(msg)
            log.debug("tags: %s, command: %s, source: %s, target: %s, "
                      "arguments: %s", tags, command, source, target, msg)
            event = Event3(command, source, target, msg, tags=tags)
            self._handle_event(event)
            if command == "ctcp" and msg[0] == "ACTION":
                event = Event3("action", source, target, msg[1:], tags=tags)
                self._handle_event(event)
        else:
            log.debug("tags: %s, command: %s, source: %s, target: %s, "
                      "arguments: %s", tags, command, source, target, [msg])
            event = Event3(command, source, target, [msg], tags=tags)
            self._handle_event(event)
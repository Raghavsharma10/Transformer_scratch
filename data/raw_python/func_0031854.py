def _process_line(self, line):
        """Process the given line and handle the events

        :param line: the raw message
        :type line: :class:`str`
        :returns: None
        :rtype: None
        :raises: None
        """
        m = self._rfc_1459_command_regexp.match(line)
        prefix = m.group('prefix')
        tags = self._process_tags(m.group('tags'))
        source = self._process_prefix(prefix)
        command = self._process_command(m.group('command'))
        arguments = self._process_arguments(m.group('argument'))
        if not self.real_server_name:
            self.real_server_name = prefix

        # Translate numerics into more readable strings.
        command = irc.events.numeric.get(command, command)
        if command not in ["privmsg", "notice"]:
            return super(ServerConnection3, self)._process_line(line)

        event = Event3("all_raw_messages", self.get_server_name(),
                       None, [line], tags=tags)
        self._handle_event(event)

        target, msg = arguments[0], arguments[1]
        messages = irc.ctcp.dequote(msg)
        command = self._resolve_command(command, target)
        for m in messages:
            self._handle_message(tags, source, command, target, m)
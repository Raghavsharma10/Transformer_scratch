def _resolve_command(self, command, target):
        """Get the correct event for the command

        Only for 'privmsg' and 'notice' commands.

        :param command: The command string
        :type command: :class:`str`
        :param target: either a user or a channel
        :type target: :class:`str`
        :returns: the correct event type
        :rtype: :class:`str`
        :raises: None
        """
        if command == "privmsg":
            if irc.client.is_channel(target):
                command = "pubmsg"
        else:
            if irc.client.is_channel(target):
                command = "pubnotice"
            else:
                command = "privnotice"
        return command
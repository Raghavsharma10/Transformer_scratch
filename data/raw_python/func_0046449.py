def reply(self, incoming, user, message, prefix=None):
        """Replies to a user in a given channel or PM.

        If the specified incoming is a user, simply sends a PM to user.
        If the specified incoming is a channel, prefixes the message with the
        user's nick and sends it to the channel.

        This is specifically useful in creating responses to commands that can
        be used in either a channel or in a PM, and responding to the person
        who invoked the command.
        """
        if not isinstance(user, User):
            user = User(user)

        if isinstance(incoming, User):
            if prefix:
                self.msg(user, "%s: %s" % (user.nick, message))
            else:
                self.msg(user, message)
        else:
            if prefix is not False:
                self.msg(incoming, "%s: %s" % (user.nick, message))
            else:
                self.msg(incoming, message)
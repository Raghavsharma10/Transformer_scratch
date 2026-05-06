def kick(self, channel, nick, message=None):
        """Attempt to kick a user from a channel.

        If a message is not provided, defaults to own nick.
        """
        self.send("KICK", channel, nick, ":%s" % (message or self.user.nick))
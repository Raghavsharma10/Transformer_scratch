def message_channel(self, message):
        """
        Nicer shortcut for sending a message to a channel. Also
        irclib doesn't handle unicode so we bypass its
        privmsg -> send_raw methods and use its socket directly.
        """
        data = "PRIVMSG %s :%s\r\n" % (self.channel, message)
        self.connection.socket.send(data.encode("utf-8"))
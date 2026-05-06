def on_pubmsg(self, connection, event):
        """
        Got a message from a channel
        """
        sender = self.get_nick(event.source)
        channel = event.target
        message = event.arguments[0]

        self.handle_message(connection, sender, channel, message)
def emit_message(self, message):
        """
        Send a message to the channel. We also emit the message
        back to the sender's WebSocket.
        """
        try:
            nickname_color = self.nicknames[self.nickname]
        except KeyError:
            # Only accept messages if we've joined.
            return
        message = message[:settings.MAX_MESSAGE_LENGTH]
        # Handle IRC commands.
        if message.startswith("/"):
            self.connection.send_raw(message.lstrip("/"))
            return
        self.message_channel(message)
        self.namespace.emit("message", self.nickname, message, nickname_color)
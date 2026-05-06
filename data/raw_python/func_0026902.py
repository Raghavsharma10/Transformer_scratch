def on_pubmsg(self, connection, event):
        """
        Messages received in the channel - send them to the WebSocket.
        """
        for message in event.arguments():
            nickname = self.get_nickname(event)
            nickname_color = self.nicknames[nickname]
            self.namespace.emit("message", nickname, message, nickname_color)
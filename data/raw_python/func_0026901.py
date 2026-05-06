def on_quit(self, connection, event):
        """
        Someone left the channel - send the nicknames list to the
        WebSocket.
        """
        nickname = self.get_nickname(event)
        nickname_color = self.nicknames[nickname]
        del self.nicknames[nickname]
        self.namespace.emit("message", nickname, "leaves", nickname_color)
        self.emit_nicknames()
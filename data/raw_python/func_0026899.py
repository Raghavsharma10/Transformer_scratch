def on_join(self, connection, event):
        """
        Someone joined the channel - send the nicknames list to the
        WebSocket.
        """
        #from time import sleep; sleep(10)  # Simulate a slow connection
        nickname = self.get_nickname(event)
        nickname_color = color(nickname)
        self.nicknames[nickname] = nickname_color
        self.namespace.emit("join")
        self.namespace.emit("message", nickname, "joins", nickname_color)
        self.emit_nicknames()
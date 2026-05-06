def on_namreply(self, connection, event):
        """
        Initial list of nicknames received - remove op/voice prefixes,
        and send the list to the WebSocket.
        """
        for nickname in event.arguments()[-1].split():
            nickname = nickname.lstrip("@+")
            self.nicknames[nickname] = color(nickname)
        self.emit_nicknames()
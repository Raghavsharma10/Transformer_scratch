def on_nick(self, connection, event):
        """
        Someone changed their nickname - send the nicknames list to the
        WebSocket.
        """
        old_nickname = self.get_nickname(event)
        old_color = self.nicknames.pop(old_nickname)
        new_nickname = event.target()
        message = "is now known as %s" % new_nickname
        self.namespace.emit("message", old_nickname, message, old_color)
        new_color = color(new_nickname)
        self.nicknames[new_nickname] = new_color
        self.emit_nicknames()
        if self.nickname == old_nickname:
            self.nickname = new_nickname
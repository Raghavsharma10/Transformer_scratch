def emit_nicknames(self):
        """
        Send the nickname list to the Websocket. Called whenever the
        nicknames list changes.
        """
        nicknames = [{"nickname": name, "color": color(name)}
                     for name in sorted(self.nicknames.keys())]
        self.namespace.emit("nicknames", nicknames)
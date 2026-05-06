def send(self, *args):
        """Sends a single raw message to the IRC server.

        Arguments are automatically joined by spaces. No newlines are allowed.
        """
        msg = " ".join(a.nick if isinstance(a, User) else str(a) for a in args)
        if "\n" in msg:
            raise ValueError("Cannot send() a newline. Args: %s" % repr(args))
        _log.debug("%s <-- %s", self.server.host, msg)
        self.socket.send(msg + "\r\n")
def handle(self):
        """
            Creates a child process that is fully controlled by this
            request handler, and serves data to and from it via the
            protocol handler.
        """
        pid, fd = pty.fork()
        if pid:
            protocol = TelnetServerProtocolHandler(self.request, fd)
            protocol.handle()
        else:
            self.execute()
def start_pty_request(self, channel, term, modes):
        """Start a PTY - intended to run it a (green)thread."""
        request = self.dummy_request()
        request._sock = channel
        request.modes = modes
        request.term = term
        request.username = self.username

        # This should block until the user quits the pty
        self.pty_handler(request, self.client_address, self.tcp_server, self.vfs, self.session)

        # Shutdown the entire session
        self.transport.close()
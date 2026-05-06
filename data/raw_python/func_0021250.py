def _send(self):
        """ Send all queued messages to the server.
        """
        data = self.output_buffer.view()
        if not data:
            return
        if self.closed():
            raise self.Error("Failed to write to closed connection {!r}".format(self.server.address))
        if self.defunct():
            raise self.Error("Failed to write to defunct connection {!r}".format(self.server.address))
        self.socket.sendall(data)
        self.output_buffer.clear()
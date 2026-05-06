def add_socket(self, socket):
        """
        Add a socket to the multiplexer.

        :param socket: The socket. If it was added already, it won't be added a
            second time.
        """
        if socket not in self._sockets:
            self._sockets.add(socket)
            socket.on_closed.connect(self.remove_socket)
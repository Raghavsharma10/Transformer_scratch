def remove_socket(self, socket):
        """
        Remove a socket from the multiplexer.

        :param socket: The socket. If it was removed already or if it wasn't
            added, the call does nothing.
        """
        if socket in self._sockets:
            socket.on_closed.disconnect(self.remove_socket)
            self._sockets.remove(socket)
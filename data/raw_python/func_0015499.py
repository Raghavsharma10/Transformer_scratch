def _read_socket(self, size):
        """
        Reads data from socket.

        :param size: Size in bytes to be read.
        :return: Data from socket
        """
        value = b''
        while len(value) < size:
            data = self.connection.recv(size - len(value))
            if not data:
                break
            value += data

        # If we got less data than we requested, the server disconnected.
        if len(value) < size:
            raise socket.error()

        return value
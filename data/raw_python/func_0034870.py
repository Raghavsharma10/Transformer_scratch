def readn(self, n):
        """
        Keep receiving data until exactly `n` bytes have been read.
        """
        data = ''

        while len(data) < n:
            received = self.sock.recv(n - len(data))

            if not len(received):
                raise socket.error('no data read from socket')

            data += received

        return data
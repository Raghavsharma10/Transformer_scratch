def recv(self, bufsize, **kws):
        """Receive data from the socket. The return value is a string
        representing the data received. The amount of data may be less than the
        ammount specified by _bufsize_. """
        return Recv(self, bufsize, timeout=self._timeout, **kws)
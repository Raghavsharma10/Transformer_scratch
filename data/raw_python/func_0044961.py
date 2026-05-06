def sendall(self, data, **kws):
        """Send data to the socket. The socket must be connected to a remote
        socket. All the data is guaranteed to be sent."""
        return SendAll(self, data, timeout=self._timeout, **kws)
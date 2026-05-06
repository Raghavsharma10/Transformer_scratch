def send(self, data, **kws):
        """Send data to the socket. The socket must be connected to a remote
        socket. Ammount sent may be less than the data provided."""
        return yield_(Send(self, data, timeout=self._timeout, **kws))
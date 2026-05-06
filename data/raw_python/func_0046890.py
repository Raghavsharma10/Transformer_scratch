def forward_tcp(self, host, port):
        """Open a connection to host:port via an ssh tunnel.

        Args:
            host (str): The host to connect to.
            port (int): The port to connect to.

        Returns:
            A socket-like object that is connected to the provided host:port.

        """

        return self.transport.open_channel(
            'direct-tcpip',
            (host, port),
            self.transport.getpeername()
        )
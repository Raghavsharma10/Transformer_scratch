def get_connection(self, address):
        """Create or retrieve a muxed connection

        Arguments:
        address -- a peer endpoint in IPv4/v6 address format; None refers
                   to the connection for unknown peers

        Return:
        a bound, connected datagram socket instance
        """

        if self.connections.has_key(address):
            return self.connections[address]
        
        # We need a new datagram socket on a dynamically assigned ephemeral port
        conn = socket.socket(self._forwarding_socket.family,
                             self._forwarding_socket.type,
                             self._forwarding_socket.proto)
        conn.bind((self._forwarding_socket.getsockname()[0], 0))
        conn.connect(self._forwarding_socket.getsockname())
        if not address:
            conn.setblocking(0)
        self.connections[address] = conn
        _logger.debug("Created new connection for address: %s", address)
        return conn
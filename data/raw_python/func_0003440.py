def get_connection(self, address):
        """Create or retrieve a muxed connection

        Arguments:
        address -- a peer endpoint in IPv4/v6 address format; None refers
                   to the connection for unknown peers

        Return:
        a bound, connected datagram socket instance, or the root socket
        in case address was None
        """

        if not address:
            return self._datagram_socket

        # Create a new datagram socket bound to the same interface and port as
        # the root socket, but connected to the given peer
        conn = socket.socket(self._datagram_socket.family,
                             self._datagram_socket.type,
                             self._datagram_socket.proto)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        conn.bind(self._datagram_socket.getsockname())
        conn.connect(address)
        _logger.debug("Created new connection for address: %s", address)
        return conn
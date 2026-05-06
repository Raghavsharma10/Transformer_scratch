def forward(self):
        """Forward a stored datagram

        When the service method returns the address of a new peer, it holds
        the datagram from that peer in this instance. In this case, this
        method will perform the forwarding step. The target connection is the
        one associated with address None if get_connection has not been called
        since the service method returned the new peer's address, and the
        connection associated with the new peer's address if it has.
        """

        assert self.payload
        assert self.payload_peer_address
        if self.connections.has_key(self.payload_peer_address):
            conn = self.connections[self.payload_peer_address]
            default = False
        else:
            conn = self.connections[None]  # propagate exception if not created
            default = True
        _logger.debug("Forwarding datagram from peer: %s, default: %s",
                      self.payload_peer_address, default)
        self._forwarding_socket.sendto(self.payload, conn.getsockname())
        self.payload = ""
        self.payload_peer_address = None
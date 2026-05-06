def service(self):
        """Service the root socket

        Read from the root socket and forward one datagram to a
        connection. The call will return without forwarding data
        if any of the following occurs:

          * An error is encountered while reading from the root socket
          * Reading from the root socket times out
          * The root socket is non-blocking and has no data available
          * An empty payload is received
          * A non-empty payload is received from an unknown peer (a peer
            for which get_connection has not yet been called); in this case,
            the payload is held by this instance and will be forwarded when
            the forward method is called

        Return:
        if the datagram received was from a new peer, then the peer's
        address; otherwise None
        """

        self.payload, self.payload_peer_address = \
          self.datagram_socket.recvfrom(UDP_MAX_DGRAM_LENGTH)
        _logger.debug("Received datagram from peer: %s",
                      self.payload_peer_address)
        if not self.payload:
            self.payload_peer_address = None
            return
        if self.connections.has_key(self.payload_peer_address):
            self.forward()
        else:
            return self.payload_peer_address
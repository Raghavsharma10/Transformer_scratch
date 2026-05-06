def connect(self, peer_address):
        """Client-side UDP connection establishment

        This method connects this object's underlying socket. It subsequently
        performs a handshake if do_handshake_on_connect was set during
        initialization.

        Arguments:
        peer_address - address tuple of server peer
        """

        self._sock.connect(peer_address)
        peer_address = self._sock.getpeername()  # substituted host addrinfo
        BIO_dgram_set_connected(self._wbio.value, peer_address)
        assert self._wbio is self._rbio
        if self._do_handshake_on_connect:
            self.do_handshake()
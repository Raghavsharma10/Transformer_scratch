def accept(self):
        """Server-side UDP connection establishment

        This method returns a server-side SSLConnection object, connected to
        that peer most recently returned from the listen method and not yet
        connected. If there is no such peer, then the listen method is invoked.

        Return value: SSLConnection connected to a new peer, None if packet
        forwarding only to an existing peer occurred.
        """

        if not self._pending_peer_address:
            if not self.listen():
                _logger.debug("Accept returning without connection")
                return
        new_conn = SSLConnection(self, self._keyfile, self._certfile, True,
                                 self._cert_reqs, self._ssl_version,
                                 self._ca_certs, self._do_handshake_on_connect,
                                 self._suppress_ragged_eofs, self._ciphers,
                                 cb_user_config_ssl_ctx=self._user_config_ssl_ctx,
                                 cb_user_config_ssl=self._user_config_ssl)
        new_peer = self._pending_peer_address
        self._pending_peer_address = None
        if self._do_handshake_on_connect:
            # Note that since that connection's socket was just created in its
            # constructor, the following operation must be blocking; hence
            # handshake-on-connect can only be used with a routing demux if
            # listen is serviced by a separate application thread, or else we
            # will hang in this call
            new_conn.do_handshake()
        _logger.debug("Accept returning new connection for new peer")
        return new_conn, new_peer
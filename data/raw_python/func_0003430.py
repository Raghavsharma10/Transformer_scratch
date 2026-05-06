def listen(self):
        """Server-side cookie exchange

        This method reads datagrams from the socket and initiates cookie
        exchange, upon whose successful conclusion one can then proceed to
        the accept method. Alternatively, accept can be called directly, in
        which case it will call this method. In order to prevent denial-of-
        service attacks, only a small, constant set of computing resources
        are used during the listen phase.

        On some platforms, listen must be called so that packets will be
        forwarded to accepted connections. Doing so is therefore recommened
        in all cases for portable code.

        Return value: a peer address if a datagram from a new peer was
        encountered, None if a datagram for a known peer was forwarded
        """

        if not hasattr(self, "_listening"):
            raise InvalidSocketError("listen called on non-listening socket")

        self._pending_peer_address = None
        try:
            peer_address = self._udp_demux.service()
        except socket.timeout:
            peer_address = None
        except socket.error as sock_err:
            if sock_err.errno != errno.EWOULDBLOCK:
                _logger.exception("Unexpected socket error in listen")
                raise
            peer_address = None

        if not peer_address:
            _logger.debug("Listen returning without peer")
            return

        # The demux advises that a datagram from a new peer may have arrived
        if type(peer_address) is tuple:
            # For this type of demux, the write BIO must be pointed at the peer
            BIO_dgram_set_peer(self._wbio.value, peer_address)
            self._udp_demux.forward()
            self._listening_peer_address = peer_address

        self._check_nbio()
        self._listening = True
        try:
            _logger.debug("Invoking DTLSv1_listen for ssl: %d",
                          self._ssl.raw)
            dtls_peer_address = DTLSv1_listen(self._ssl.value)
        except openssl_error() as err:
            if err.ssl_error == SSL_ERROR_WANT_READ:
                # This method must be called again to forward the next datagram
                _logger.debug("DTLSv1_listen must be resumed")
                return
            elif err.errqueue and err.errqueue[0][0] == ERR_WRONG_VERSION_NUMBER:
                _logger.debug("Wrong version number; aborting handshake")
                raise
            elif err.errqueue and err.errqueue[0][0] == ERR_COOKIE_MISMATCH:
                _logger.debug("Mismatching cookie received; aborting handshake")
                raise
            elif err.errqueue and err.errqueue[0][0] == ERR_NO_SHARED_CIPHER:
                _logger.debug("No shared cipher; aborting handshake")
                raise
            _logger.exception("Unexpected error in DTLSv1_listen")
            raise
        finally:
            self._listening = False
            self._listening_peer_address = None
        if type(peer_address) is tuple:
            _logger.debug("New local peer: %s", dtls_peer_address)
            self._pending_peer_address = peer_address
        else:
            self._pending_peer_address = dtls_peer_address
        _logger.debug("New peer: %s", self._pending_peer_address)
        return self._pending_peer_address
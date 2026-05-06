def do_handshake(self):
        """Perform a handshake with the peer

        This method forces an explicit handshake to be performed with either
        the client or server peer.
        """

        _logger.debug("Initiating handshake...")
        try:
            self._wrap_socket_library_call(
                lambda: SSL_do_handshake(self._ssl.value),
                ERR_HANDSHAKE_TIMEOUT)
        except openssl_error() as err:
            if err.ssl_error == SSL_ERROR_SYSCALL and err.result == -1:
                raise_ssl_error(ERR_PORT_UNREACHABLE, err)
            raise
        self._handshake_done = True
        _logger.debug("...completed handshake")
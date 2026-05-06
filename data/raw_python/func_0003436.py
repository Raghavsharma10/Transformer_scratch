def shutdown(self):
        """Shut down the DTLS connection

        This method attemps to complete a bidirectional shutdown between
        peers. For non-blocking sockets, it should be called repeatedly until
        it no longer raises continuation request exceptions.
        """

        if hasattr(self, "_listening"):
            # Listening server-side sockets cannot be shut down
            return

        try:
            self._wrap_socket_library_call(
                lambda: SSL_shutdown(self._ssl.value), ERR_READ_TIMEOUT)
        except openssl_error() as err:
            if err.result == 0:
                # close-notify alert was just sent; wait for same from peer
                # Note: while it might seem wise to suppress further read-aheads
                # with SSL_set_read_ahead here, doing so causes a shutdown
                # failure (ret: -1, SSL_ERROR_SYSCALL) on the DTLS shutdown
                # initiator side. And test_starttls does pass.
                self._wrap_socket_library_call(
                    lambda: SSL_shutdown(self._ssl.value), ERR_READ_TIMEOUT)
            else:
                raise
        if hasattr(self, "_rsock"):
            # Return wrapped connected server socket (non-listening)
            return _UnwrappedSocket(self._sock, self._rsock, self._udp_demux,
                                    self._ctx,
                                    BIO_dgram_get_peer(self._wbio.value))
        # Return unwrapped client-side socket or unwrapped server-side socket
        # for single-socket servers
        return self._sock
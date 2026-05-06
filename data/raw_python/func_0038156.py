def connect(self, addr):
        """Connects to remote ADDR, and then wraps the connection in
        an SSL channel."""
        # Here we assume that the socket is client-side, and not
        # connected at the time of the call.  We connect it, then wrap it.
        if self._sslobj:
            raise ValueError("attempt to connect already-connected SSLSocket!")
        socket.connect(self, addr)
        if six.PY3:
            self._sslobj = self.context._wrap_socket(self._sock, False, self.server_hostname)
        else:
            if self.ciphers is None:
                self._sslobj = _ssl.sslwrap(self._sock, False, self.keyfile, self.certfile,
                                            self.cert_reqs, self.ssl_version,
                                            self.ca_certs)
            else:
                self._sslobj = _ssl.sslwrap(self._sock, False, self.keyfile, self.certfile,
                                            self.cert_reqs, self.ssl_version,
                                            self.ca_certs, self.ciphers)
        if self.do_handshake_on_connect:
            self.do_handshake()
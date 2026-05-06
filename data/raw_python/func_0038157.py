def accept(self):
        """Accepts a new connection from a remote client, and returns
        a tuple containing that new connection wrapped with a server-side
        SSL channel, and the address of the remote client."""
        newsock, addr = socket.accept(self)
        ssl_sock = SSLSocket(newsock._sock,
                             keyfile=self.keyfile,
                             certfile=self.certfile,
                             server_side=True,
                             cert_reqs=self.cert_reqs,
                             ssl_version=self.ssl_version,
                             ca_certs=self.ca_certs,
                             do_handshake_on_connect=self.do_handshake_on_connect,
                             suppress_ragged_eofs=self.suppress_ragged_eofs,
                             ciphers=self.ciphers)
        return ssl_sock, addr
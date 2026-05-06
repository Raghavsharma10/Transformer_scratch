def _new_conn(self):
        """
        Return a fresh :class:`httplib.HTTPSConnection`.
        """
        self.num_connections += 1
        log.info("Starting new HTTPS connection (%d): %s"
                 % (self.num_connections, self.host))

        if not self.ConnectionCls or self.ConnectionCls is DummyConnection:
            # Platform-specific: Python without ssl
            raise SSLError("Can't connect to HTTPS URL because the SSL "
                           "module is not available.")

        actual_host = self.host
        actual_port = self.port
        if self.proxy is not None:
            actual_host = self.proxy.host
            actual_port = self.proxy.port

        conn = self.ConnectionCls(host=actual_host, port=actual_port,
                                  timeout=self.timeout.connect_timeout,
                                  strict=self.strict, **self.conn_kw)

        return self._prepare_conn(conn)
def connect(self, host, port):
        '''Connect to the provided host, port'''
        conn = connection.Connection(host, port,
            reconnection_backoff=self._reconnection_backoff,
            auth_secret=self._auth_secret,
            timeout=self._connect_timeout,
            **self._identify_options)
        if conn.alive():
            conn.setblocking(0)
        self.add(conn)
        return conn
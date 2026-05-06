def connect(self):
        """Connect to the Redis server or Cluster.

        :rtype: tornado.concurrent.Future

        """
        LOGGER.debug('Creating a%s connection to %s:%s (db %s)',
                     ' cluster node'
                     if self._clustering else '', self._hosts[0]['host'],
                     self._hosts[0]['port'], self._hosts[0].get(
                         'db', DEFAULT_DB))
        self._connect_future = concurrent.Future()
        conn = _Connection(
            self._hosts[0]['host'],
            self._hosts[0]['port'],
            self._hosts[0].get('db', DEFAULT_DB),
            self._read,
            self._on_closed,
            self.io_loop,
            cluster_node=self._clustering)
        self.io_loop.add_future(conn.connect(), self._on_connected)
        return self._connect_future
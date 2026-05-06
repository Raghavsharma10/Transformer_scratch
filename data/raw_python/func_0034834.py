def close(self):
        """Close any open connections to Redis.

        :raises: :exc:`tredis.exceptions.ConnectionError`

        """
        if not self._connected.is_set():
            raise exceptions.ConnectionError('not connected')
        self._closing = True
        if self._clustering:
            for host in self._cluster.keys():
                self._cluster[host].close()
        elif self._connection:
            self._connection.close()
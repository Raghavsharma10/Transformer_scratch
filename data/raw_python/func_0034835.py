def ready(self):
        """Indicates that the client is connected to the Redis server or
        cluster and is ready for use.

        :rtype: bool

        """
        if self._clustering:
            return (all([c.connected for c in self._cluster.values()])
                    and len(self._cluster))
        return (self._connection and self._connection.connected)
def connection_from_host(self, host, port=None, scheme='http'):
        """
        Get a :class:`ConnectionPool` based on the host, port, and scheme.

        If ``port`` isn't given, it will be derived from the ``scheme`` using
        ``urllib3.connectionpool.port_by_scheme``.
        """

        if not host:
            raise LocationValueError("No host specified.")

        scheme = scheme or 'http'
        port = port or port_by_scheme.get(scheme, 80)
        pool_key = (scheme, host, port)

        with self.pools.lock:
            # If the scheme, host, or port doesn't match existing open
            # connections, open a new ConnectionPool.
            pool = self.pools.get(pool_key)
            if pool:
                return pool

            # Make a fresh ConnectionPool of the desired type
            pool = self._new_pool(scheme, host, port)
            self.pools[pool_key] = pool

        return pool
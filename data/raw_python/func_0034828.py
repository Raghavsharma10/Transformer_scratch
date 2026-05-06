def connect(self):
        """Connect to the Redis server if necessary.

        :rtype: :class:`~tornado.concurrent.Future`
        :raises: :class:`~tredis.exceptions.ConnectError`
                 :class:`~tredis.exceptinos.RedisError`

        """
        future = concurrent.Future()

        if self.connected:
            raise exceptions.ConnectError('already connected')

        LOGGER.debug('%s connecting', self.name)
        self.io_loop.add_future(
            self._client.connect(self.host, self.port),
            lambda f: self._on_connected(f, future))
        return future
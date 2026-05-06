def connect(self):
        """
        Connect to the controller and start processing responses.
        """
        logger.info("Connecting to %s:%s", self._host, self._port)
        reader, writer = yield from asyncio.open_connection(
                self._host, self._port, loop=self._loop)
        self._ioloop_future = ensure_future(
                self._ioloop(reader, writer), loop=self._loop)
        logger.info("Connected")
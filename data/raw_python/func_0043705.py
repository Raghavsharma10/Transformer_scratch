def close(self):
        """
        Disconnect from the controller.
        """
        logger.info("Closing connection to %s:%s", self._host, self._port)
        self._ioloop_future.cancel()
        try:
            yield from self._ioloop_future
        except asyncio.CancelledError:
            pass
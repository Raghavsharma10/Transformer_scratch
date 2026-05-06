def open_channel(self):
        """
        Open a new channel on this connection.

        This method is a :ref:`coroutine <coroutine>`.

        :return: The new :class:`Channel` object.
        """
        if self._closing:
            raise ConnectionClosed("Closed by application")
        if self.closed.done():
            raise self.closed.exception()

        channel = yield from self.channel_factory.open()
        return channel
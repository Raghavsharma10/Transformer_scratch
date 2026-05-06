def _on_connected(self, stream_future, connect_future):
        """Invoked when the socket stream has connected, setting up the
        stream callbacks and invoking the on connect callback if set.

        :param stream_future: The connection socket future
        :type stream_future: :class:`~tornado.concurrent.Future`
        :param stream_future: The connection response future
        :type stream_future: :class:`~tornado.concurrent.Future`
        :raises: :exc:`tredis.exceptions.ConnectError`

        """
        if stream_future.exception():
            connect_future.set_exception(
                exceptions.ConnectError(stream_future.exception()))
        else:
            self._stream = stream_future.result()
            self._stream.set_close_callback(self._on_closed)
            self.connected = True
            connect_future.set_result(self)
def length(self):
        """
        Gives the length of the queue. Returns ``None`` if the queue is not
        connected.

        If the queue is not connected then it will raise
        :class:`retask.ConnectionError`.

        """
        if not self.connected:
            raise ConnectionError('Queue is not connected')

        try:
            length = self.rdb.llen(self._name)
        except redis.exceptions.ConnectionError as err:
            raise ConnectionError(str(err))

        return length
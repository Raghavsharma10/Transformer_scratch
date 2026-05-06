def names(self):
        """
        Returns a list of queues available, ``None`` if no such
        queues found. Remember this will only shows queues with
        at least one item enqueued.
        """
        data = None
        if not self.connected:
            raise ConnectionError('Queue is not connected')

        try:
            data = self.rdb.keys("retaskqueue-*")
        except redis.exceptions.ConnectionError as err:
            raise ConnectionError(str(err))

        return [name[12:] for name in data]
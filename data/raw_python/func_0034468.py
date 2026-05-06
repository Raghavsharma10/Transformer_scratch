def pexpireat(self, key, timestamp):
        """:meth:`~tredis.RedisClient.pexpireat` has the same effect and
        semantic as :meth:`~tredis.RedisClient.expireat`, but the Unix time
        at which the key will expire is specified in milliseconds instead of
        seconds.

        .. note::

           **Time complexity**: ``O(1)``

        :param key: The key to set an expiration for
        :type key: :class:`str`, :class:`bytes`
        :param int timestamp: The expiration UNIX epoch value in milliseconds
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute(
            [b'PEXPIREAT', key,
             ascii(timestamp).encode('ascii')], 1)
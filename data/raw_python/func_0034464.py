def expireat(self, key, timestamp):
        """:meth:`~tredis.RedisClient.expireat` has the same effect and
        semantic as :meth:`~tredis.RedisClient.expire`, but instead of
        specifying the number of seconds representing the TTL (time to live),
        it takes an absolute Unix timestamp (seconds since January 1, 1970).

        Please for the specific semantics of the command refer to the
        documentation of :meth:`~tredis.RedisClient.expire`.

        .. note::

           **Time complexity**: ``O(1)``

        :param key: The key to set an expiration for
        :type key: :class:`str`, :class:`bytes`
        :param int timestamp: The UNIX epoch value for the expiration
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute(
            [b'EXPIREAT', key,
             ascii(timestamp).encode('ascii')], 1)
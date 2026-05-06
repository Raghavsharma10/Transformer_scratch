def setex(self, key, seconds, value):
        """Set key to hold the string value and set key to timeout after a
        given number of seconds.

        :meth:`~tredis.RedisClient.setex` is atomic, and can be reproduced by
        using :meth:`~tredis.RedisClient.set` and
        :meth:`~tredis.RedisClient.expire` inside an
        :meth:`~tredis.RedisClient.multi` /
        :meth:`~tredis.RedisClient.exec` block. It is provided as a faster
        alternative to the given sequence of operations, because this operation
        is very common when Redis is used as a cache.

        An error is returned when seconds is invalid.

        .. versionadded:: 0.2.0

        .. note:: **Time complexity**: ``O(1)``

        :param key: The key to set
        :type key: :class:`str`, :class:`bytes`
        :param int seconds: Number of seconds for TTL
        :param value: The value to set
        :type value: :class:`str`, :class:`bytes`
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute([b'SETEX', key, ascii(seconds), value], b'OK')
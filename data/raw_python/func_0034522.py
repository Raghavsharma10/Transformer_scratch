def psetex(self, key, milliseconds, value):
        """:meth:`~tredis.RedisClient.psetex` works exactly like
        :meth:`~tredis.RedisClient.psetex` with the sole difference that the
        expire time is specified in milliseconds instead of seconds.

        .. versionadded:: 0.2.0

        .. note:: **Time complexity**: ``O(1)``

        :param key: The key to set
        :type key: :class:`str`, :class:`bytes`
        :param int milliseconds: Number of milliseconds for TTL
        :param value: The value to set
        :type value: :class:`str`, :class:`bytes`
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute(
            [b'PSETEX', key, ascii(milliseconds), value], b'OK')
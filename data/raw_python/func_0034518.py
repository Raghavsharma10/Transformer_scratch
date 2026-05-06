def incrby(self, key, increment):
        """Increments the number stored at key by increment. If the key does
        not exist, it is set to 0 before performing the operation. An error is
        returned if the key contains a value of the wrong type or contains a
        string that can not be represented as integer. This operation is
        limited to 64 bit signed integers.

        See :meth:`~tredis.RedisClient.incr` for extra information on
        increment/decrement operations.

        .. versionadded:: 0.2.0

        .. note:: **Time complexity**: ``O(1)``

        :param key: The key to increment
        :type key: :class:`str`, :class:`bytes`
        :param int increment: The amount to increment by
        :returns: The value of key after the increment
        :rtype: int
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute([b'INCRBY', key, ascii(increment)])
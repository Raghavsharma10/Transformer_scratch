def decrby(self, key, decrement):
        """Decrements the number stored at key by decrement. If the key does
        not exist, it is set to 0 before performing the operation. An error
        is returned if the key contains a value of the wrong type or contains
        a string that can not be represented as integer. This operation is
        limited to 64 bit signed integers.

        See :meth:`~tredis.RedisClient.incr` for extra information on
        increment/decrement operations.

        .. versionadded:: 0.2.0

        .. note:: **Time complexity**: ``O(1)``

        :param key: The key to decrement
        :type key: :class:`str`, :class:`bytes`
        :param int decrement: The amount to decrement by
        :returns: The value of key after the decrement
        :rtype: int
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute([b'DECRBY', key, ascii(decrement)])
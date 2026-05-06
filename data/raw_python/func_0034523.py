def set(self, key, value, ex=None, px=None, nx=False, xx=False):
        """Set key to hold the string value. If key already holds a value, it
        is overwritten, regardless of its type. Any previous time to live
        associated with the key is discarded on successful
        :meth:`~tredis.RedisClient.set` operation.

        If the value is not one of :class:`str`, :class:`bytes`, or
        :class:`int`, a :exc:`ValueError` will be raised.

        .. note:: **Time complexity**: ``O(1)``

        :param key: The key to remove
        :type key: :class:`str`, :class:`bytes`
        :param value: The value to set
        :type value: :class:`str`, :class:`bytes`, :class:`int`
        :param int ex: Set the specified expire time, in seconds
        :param int px: Set the specified expire time, in milliseconds
        :param bool nx: Only set the key if it does not already exist
        :param bool xx: Only set the key if it already exist
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`
        :raises: :exc:`ValueError`

        """
        command = [b'SET', key, value]
        if ex:
            command += [b'EX', ascii(ex).encode('ascii')]
        if px:
            command += [b'PX', ascii(px).encode('ascii')]
        if nx:
            command.append(b'NX')
        if xx:
            command.append(b'XX')
        return self._execute(command, b'OK')
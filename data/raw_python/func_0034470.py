def restore(self, key, ttl, value, replace=False):
        """Create a key associated with a value that is obtained by
        deserializing the provided serialized value (obtained via
        :meth:`~tredis.RedisClient.dump`).

        If ``ttl`` is ``0`` the key is created without any expire, otherwise
        the specified expire time (in milliseconds) is set.

        :meth:`~tredis.RedisClient.restore` will return a
        ``Target key name is busy`` error when key already exists unless you
        use the :meth:`~tredis.RedisClient.restore` modifier (Redis 3.0 or
        greater).

        :meth:`~tredis.RedisClient.restore` checks the RDB version and data
        checksum. If they don't match an error is returned.

        .. note::

           **Time complexity**: ``O(1)`` to create the new key and additional
           ``O(N*M)`` to reconstruct the serialized value, where ``N`` is the
           number of Redis objects composing the value and ``M`` their average
           size. For small string values the time complexity is thus
           ``O(1)+O(1*M)`` where ``M`` is small, so simply ``O(1)``. However
           for sorted set values the complexity is ``O(N*M*log(N))`` because
           inserting values into sorted sets is ``O(log(N))``.

        :param key: The key to get the TTL for
        :type key: :class:`str`, :class:`bytes`
        :param int ttl: The number of seconds to set the timeout to
        :param value: The value to restore to the key
        :type value: :class:`str`, :class:`bytes`
        :param bool replace: Replace a pre-existing key
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        command = [b'RESTORE', key, ascii(ttl).encode('ascii'), value]
        if replace:
            command.append(b'REPLACE')
        return self._execute(command, b'OK')
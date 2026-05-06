def mset(self, mapping):
        """Sets the given keys to their respective values.
        :meth:`~tredis.RedisClient.mset` replaces existing values with new
        values, just as regular :meth:`~tredis.RedisClient.set`. See
        :meth:`~tredis.RedisClient.msetnx` if you don't want to overwrite
        existing values.

        :meth:`~tredis.RedisClient.mset` is atomic, so all given keys are set
        at once. It is not possible for clients to see that some of the keys
        were updated while others are unchanged.

        .. versionadded:: 0.2.0

        .. note:: **Time complexity**: ``O(N)`` where ``N`` is the number of
           keys to set.

        :param dict mapping: A mapping of key/value pairs to set
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        command = [b'MSET']
        for key, value in mapping.items():
            command += [key, value]
        return self._execute(command, b'OK')
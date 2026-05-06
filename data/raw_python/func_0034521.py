def msetnx(self, mapping):
        """Sets the given keys to their respective values.
        :meth:`~tredis.RedisClient.msetnx` will not perform any operation at
        all even if just a single key already exists.

        Because of this semantic :meth:`~tredis.RedisClient.msetnx` can be used
        in order to set different keys representing different fields of an
        unique logic object in a way that ensures that either all the fields or
        none at all are set.

        :meth:`~tredis.RedisClient.msetnx` is atomic, so all given keys are set
        at once. It is not possible for clients to see that some of the keys
        were updated while others are unchanged.

        .. versionadded:: 0.2.0

        .. note:: **Time complexity**: ``O(N)`` where ``N`` is the number of
           keys to set.

        :param dict mapping: A mapping of key/value pairs to set
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        command = [b'MSETNX']
        for key, value in mapping.items():
            command += [key, value]
        return self._execute(command, 1)
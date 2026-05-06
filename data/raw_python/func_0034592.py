def srandmember(self, key, count=None):
        """When called with just the key argument, return a random element from
        the set value stored at key.

        Starting from Redis version 2.6, when called with the additional count
        argument, return an array of count distinct elements if count is
        positive. If called with a negative count the behavior changes and the
        command is allowed to return the same element multiple times. In this
        case the number of returned elements is the absolute value of the
        specified count.

        When called with just the key argument, the operation is similar to
        :meth:`~tredis.RedisClient.spop`, however while
        :meth:`~tredis.RedisClient.spop` also removes the randomly selected
        element from the set, :meth:`~tredis.RedisClient.srandmember` will just
        return a random element without altering the original set in any way.

        .. note::

           **Time complexity**: Without the count argument ``O(1)``, otherwise
           ``O(N)`` where ``N`` is the absolute value of the passed count.

        :param key: The key to get one or more random members from
        :type key: :class:`str`, :class:`bytes`
        :param int count: The number of members to return
        :rtype: bytes, list
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        command = [b'SRANDMEMBER', key]
        if count:
            command.append(ascii(count).encode('ascii'))
        return self._execute(command)
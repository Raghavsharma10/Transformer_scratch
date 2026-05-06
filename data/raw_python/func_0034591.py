def spop(self, key, count=None):
        """Removes and returns one or more random elements from the set value
        store at key.

        This operation is similar to :meth:`~tredis.RedisClient.srandmember`,
        that returns one or more random elements from a set but does not remove
        it.

        The count argument will be available in a later version and is not
        available in 2.6, 2.8, 3.0

        Redis 3.2 will be the first version where an optional count argument
        can be passed to :meth:`~tredis.RedisClient.spop` in order to retrieve
        multiple elements in a single call. The implementation is already
        available in the unstable branch.

        .. note::

           **Time complexity**: Without the count argument ``O(1)``, otherwise
           ``O(N)`` where ``N`` is the absolute value of the passed count.

        :param key: The key to get one or more random members from
        :type key: :class:`str`, :class:`bytes`
        :param int count: The number of members to return
        :rtype: bytes, list
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        command = [b'SPOP', key]
        if count:  # pragma: nocover
            command.append(ascii(count).encode('ascii'))
        return self._execute(command)
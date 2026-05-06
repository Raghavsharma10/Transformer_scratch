def sinterstore(self, destination, *keys):
        """This command is equal to :meth:`~tredis.RedisClient.sinter`, but
        instead of returning the resulting set, it is stored in destination.

        If destination already exists, it is overwritten.

        .. note::

           **Time complexity**: ``O(N*M)`` worst case where ``N`` is the
           cardinality of the smallest set and ``M`` is the number of sets.

        :param destination: The set to store the intersection into
        :type destination: :class:`str`, :class:`bytes`
        :param keys: One or more set keys as positional arguments
        :type keys: :class:`str`, :class:`bytes`
        :rtype: int
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute([b'SINTERSTORE', destination] + list(keys))
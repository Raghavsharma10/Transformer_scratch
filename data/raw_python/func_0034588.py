def sdiffstore(self, destination, *keys):
        """This command is equal to :meth:`~tredis.RedisClient.sdiff`, but
        instead of returning the resulting set, it is stored in destination.

        If destination already exists, it is overwritten.

        .. note::

           **Time complexity**: ``O(N)`` where ``N`` is the total number of
           elements in all given sets.

        :param destination: The set to store the diff into
        :type destination: :class:`str`, :class:`bytes`
        :param keys: One or more set keys as positional arguments
        :type keys: :class:`str`, :class:`bytes`
        :rtype: int
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute([b'SDIFFSTORE', destination] + list(keys))
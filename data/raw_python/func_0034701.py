def pfmerge(self, dest_key, *keys):
        """Merge multiple HyperLogLog values into an unique value that will
        approximate the cardinality of the union of the observed Sets of the
        source HyperLogLog structures.

        The computed merged HyperLogLog is set to the destination variable,
        which is created if does not exist (defaulting to an empty
        HyperLogLog).

        .. versionadded:: 0.2.0

        .. note::

           **Time complexity**: ``O(N)`` to merge ``N`` HyperLogLogs, but
           with high constant times.

        :param dest_key: The destination key
        :type dest_key: :class:`str`, :class:`bytes`
        :param keys: One or more keys
        :type keys: :class:`str`, :class:`bytes`
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute([b'PFMERGE', dest_key] + list(keys), b'OK')
def srem(self, key, *members):
        """Remove the specified members from the set stored at key. Specified
        members that are not a member of this set are ignored. If key does not
        exist, it is treated as an empty set and this command returns ``0``.

        An error is returned when the value stored at key is not a set.

        Returns :data:`True` if all requested members are removed. If more
        than one member is passed in and not all members are removed, the
        number of removed members is returned.

        .. note::

           **Time complexity**: ``O(N)`` where ``N`` is the number of members
           to be removed.

        :param key: The key to remove the member from
        :type key: :class:`str`, :class:`bytes`
        :param mixed members: One or more member values to remove
        :rtype: bool, int
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute([b'SREM', key] + list(members), len(members))
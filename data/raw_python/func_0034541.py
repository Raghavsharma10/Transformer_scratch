def zrem(self, key, *members):
        """Removes the specified members from the sorted set stored at key.
         Non existing members are ignored.

        An error is returned when key exists and does not hold a sorted set.

        .. note::

           **Time complexity**: ``O(M*log(N))`` with ``N`` being the number of
           elements in the sorted set and ``M`` the number of elements to be
           removed.

        :param key: The key of the sorted set
        :type key: :class:`str`, :class:`bytes`
        :param members: One or more member values to remove
        :type members: :class:`str`, :class:`bytes`
        :rtype: int
        :raises: :exc:`~tredis.exceptions.RedisError`
        """
        return self._execute([b'ZREM', key] + list(members))
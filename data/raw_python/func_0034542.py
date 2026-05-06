def zremrangebyscore(self, key, min_score, max_score):
        """Removes all elements in the sorted set stored at key with a score
        between min and max.

        Intervals are described in :meth:`~tredis.RedisClient.zrangebyscore`.

        Returns the number of elements removed.

        .. note::

           **Time complexity**: ``O(log(N)+M)`` with ``N`` being the number of
           elements in the sorted set and M the number of elements removed by
           the operation.

        :param key: The key of the sorted set
        :type key: :class:`str`, :class:`bytes`
        :param min_score: Lowest score definition
        :type min_score: :class:`str`, :class:`bytes`
        :param max_score: Highest score definition
        :type max_score: :class:`str`, :class:`bytes`
        :rtype: int
        :raises: :exc:`~tredis.exceptions.RedisError`
        """
        return self._execute([b'ZREMRANGEBYSCORE', key, min_score, max_score])
def zrevrange(self, key, start=0, stop=-1, with_scores=False):
        """Returns the specified range of elements in the sorted set stored at
        key. The elements are considered to be ordered from the highest to the
        lowest score. Descending lexicographical order is used for elements
        with equal score.

        Apart from the reversed ordering, :py:meth:`~tredis.Client.zrevrange`
        is similar to :py:meth:`~tredis.Client.zrange` .

        .. note::

           **Time complexity**: ``O(log(N)+M)`` with ``N`` being the number of
           elements in the sorted set and ``M`` the number of elements
           returned.

        :param key: The key of the sorted set
        :type key: :class:`str`, :class:`bytes`
        :param int start: The starting index of the sorted set
        :param int stop: The ending index of the sorted set
        :param bool with_scores: Return the scores with the elements

        :rtype: list
        :raises: :exc:`~tredis.exceptions.RedisError`
        """
        command = [b'ZREVRANGE', key, start, stop]
        if with_scores:
            command += ['WITHSCORES']
        return self._execute(command)
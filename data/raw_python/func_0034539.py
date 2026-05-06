def zrange(self, key, start=0, stop=-1, with_scores=False):
        """Returns the specified range of elements in the sorted set stored at
        key. The elements are considered to be ordered from the lowest to the
        highest score. Lexicographical order is used for elements with equal
        score.

        See :meth:`tredis.Client.zrevrange` when you need the elements ordered
        from highest to lowest score (and descending lexicographical order for
        elements with equal score).

        Both start and stop are zero-based indexes, where ``0`` is the first
        element, ``1`` is the next element and so on. They can also be negative
        numbers indicating offsets from the end of the sorted set, with ``-1``
        being the last element of the sorted set, ``-2`` the penultimate
        element and so on.

        ``start`` and ``stop`` are inclusive ranges, so for example
        ``ZRANGE myzset 0 1`` will return both the first and the second element
        of the sorted set.

        Out of range indexes will not produce an error. If start is larger than
        the largest index in the sorted set, or ``start > stop``, an empty list
        is returned. If stop is larger than the end of the sorted set Redis
        will treat it like it is the last element of the sorted set.

        It is possible to pass the ``WITHSCORES`` option in order to return the
        scores of the elements together with the elements. The returned list
        will contain ``value1,score1,...,valueN,scoreN`` instead of
        ``value1,...,valueN``. Client libraries are free to return a more
        appropriate data type (suggestion: an array with (value, score)
        arrays/tuples).

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
        command = [b'ZRANGE', key, start, stop]
        if with_scores:
            command += ['WITHSCORES']
        return self._execute(command)
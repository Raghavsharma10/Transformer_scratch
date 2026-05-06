def bitcount(self, key, start=None, end=None):
        """Count the number of set bits (population counting) in a string.

        By default all the bytes contained in the string are examined. It is
        possible to specify the counting operation only in an interval passing
        the additional arguments start and end.

        Like for the :meth:`~tredis.RedisClient.getrange` command start and
        end can contain negative values in order to index bytes starting from
        the end of the string, where ``-1`` is the last byte, ``-2`` is the
        penultimate, and so forth.

        Non-existent keys are treated as empty strings, so the command will
        return zero.

        .. versionadded:: 0.2.0

        .. note:: **Time complexity**: ``O(N)``

        :param key: The key to get
        :type key: :class:`str`, :class:`bytes`
        :param int start: The start position to evaluate in the string
        :param int end: The end position to evaluate in the string
        :rtype: int
        :raises: :exc:`~tredis.exceptions.RedisError`, :exc:`ValueError`

        """
        command = [b'BITCOUNT', key]
        if start is not None and end is None:
            raise ValueError('Can not specify start without an end')
        elif start is None and end is not None:
            raise ValueError('Can not specify start without an end')
        elif start is not None and end is not None:
            command += [ascii(start), ascii(end)]
        return self._execute(command)
def bitpos(self, key, bit, start=None, end=None):
        """Return the position of the first bit set to ``1`` or ``0`` in a
        string.

        The position is returned, thinking of the string as an array of bits
        from left to right, where the first byte's most significant bit is at
        position 0, the second byte's most significant bit is at position
        ``8``, and so forth.

        The same bit position convention is followed by
        :meth:`~tredis.RedisClient.getbit` and
        :meth:`~tredis.RedisClient.setbit`.

        By default, all the bytes contained in the string are examined. It is
        possible to look for bits only in a specified interval passing the
        additional arguments start and end (it is possible to just pass start,
        the operation will assume that the end is the last byte of the string.
        However there are semantic differences as explained later). The range
        is interpreted as a range of bytes and not a range of bits, so
        ``start=0`` and ``end=2`` means to look at the first three bytes.

        Note that bit positions are returned always as absolute values starting
        from bit zero even when start and end are used to specify a range.

        Like for the :meth:`~tredis.RedisClient.getrange` command start and
        end can contain negative values in order to index bytes starting from
        the end of the string, where ``-1`` is the last byte, ``-2`` is the
        penultimate, and so forth.

        Non-existent keys are treated as empty strings.

        .. versionadded:: 0.2.0

        .. note:: **Time complexity**: ``O(N)``

        :param key: The key to get
        :type key: :class:`str`, :class:`bytes`
        :param int bit: The bit value to search for (``1`` or ``0``)
        :param int start: The start position to evaluate in the string
        :param int end: The end position to evaluate in the string
        :returns: The position of the first bit set to ``1`` or ``0``
        :rtype: int
        :raises: :exc:`~tredis.exceptions.RedisError`, :exc:`ValueError`

        """
        if 0 < bit > 1:
            raise ValueError('bit must be 1 or 0, not {}'.format(bit))
        command = [b'BITPOS', key, ascii(bit)]
        if start is not None and end is None:
            raise ValueError('Can not specify start without an end')
        elif start is None and end is not None:
            raise ValueError('Can not specify start without an end')
        elif start is not None and end is not None:
            command += [ascii(start), ascii(end)]
        return self._execute(command)
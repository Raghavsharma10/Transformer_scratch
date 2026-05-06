def lrange(self, key, start, end):
        """
        Returns the specified elements of the list stored at key.

        :param key: The list's key
        :type key: :class:`str`, :class:`bytes`
        :param int start: zero-based index to start retrieving elements from
        :param int end: zero-based index at which to stop retrieving elements

        :rtype: list
        :raises: :exc:`~tredis.exceptions.TRedisException`

        The offsets start and stop are zero-based indexes, with 0 being the
        first element of the list (the head of the list), 1 being the next
        element and so on.

        These offsets can also be negative numbers indicating offsets
        starting at the end of the list. For example, -1 is the last element
        of the list, -2 the penultimate, and so on.

        Note that if you have a list of numbers from 0 to 100,
        ``lrange(key, 0, 10)`` will return 11 elements, that is, the
        rightmost item is included. This may or may not be consistent with
        behavior of range-related functions in your programming language of
        choice (think Ruby's ``Range.new``, ``Array#slice`` or Python's
        :func:`range` function).

        Out of range indexes will not produce an error. If start is larger
        than the end of the list, an empty list is returned. If stop is
        larger than the actual end of the list, Redis will treat it like the
        last element of the list.

        .. note::

           **Time complexity** ``O(S+N)`` where ``S`` is the distance of
           start offset from ``HEAD`` for small lists, from nearest end
           (``HEAD`` or ``TAIL``) for large lists; and ``N`` is the number
           of elements in the specified range.

        """
        return self._execute([b'LRANGE', key, start, end])
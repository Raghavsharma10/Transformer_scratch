def getrange(self, key, start, end):
        """Returns the bit value at offset in the string value stored at key.

        When offset is beyond the string length, the string is assumed to be a
        contiguous space with 0 bits. When key does not exist it is assumed to
        be an empty string, so offset is always out of range and the value is
        also assumed to be a contiguous space with 0 bits.

        .. versionadded:: 0.2.0

        .. note:: **Time complexity**: ``O(N)`` where ``N`` is the length of
           the returned string. The complexity is ultimately determined by the
           returned length, but because creating a substring from an existing
           string is very cheap, it can be considered ``O(1)`` for small
           strings.

        :param key: The key to get the bit from
        :type key: :class:`str`, :class:`bytes`
        :param int start: The start position to evaluate in the string
        :param int end: The end position to evaluate in the string
        :rtype: bytes|None
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute([b'GETRANGE', key, ascii(start), ascii(end)])
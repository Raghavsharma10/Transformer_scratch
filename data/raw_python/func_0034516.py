def getbit(self, key, offset):
        """Returns the bit value at offset in the string value stored at key.

        When offset is beyond the string length, the string is assumed to be a
        contiguous space with 0 bits. When key does not exist it is assumed to
        be an empty string, so offset is always out of range and the value is
        also assumed to be a contiguous space with 0 bits.

        .. versionadded:: 0.2.0

        .. note:: **Time complexity**: ``O(1)``

        :param key: The key to get the bit from
        :type key: :class:`str`, :class:`bytes`
        :param int offset: The bit offset to fetch the bit from
        :rtype: bytes|None
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute([b'GETBIT', key, ascii(offset)])
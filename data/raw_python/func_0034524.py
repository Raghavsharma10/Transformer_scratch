def setbit(self, key, offset, bit):
        """Sets or clears the bit at offset in the string value stored at key.

        The bit is either set or cleared depending on value, which can be
        either 0 or 1. When key does not exist, a new string value is created.
        The string is grown to make sure it can hold a bit at offset. The
        offset argument is required to be greater than or equal to 0, and
        smaller than 2 :sup:`32` (this limits bitmaps to 512MB). When the
        string at key is grown, added bits are set to 0.

        .. warning:: When setting the last possible bit (offset equal to
           2 :sup:`32` -1) and the string value stored at key does not yet hold
           a string value, or holds a small string value, Redis needs to
           allocate all intermediate memory which can block the server for some
           time. On a 2010 MacBook Pro, setting bit number 2 :sup:`32` -1
           (512MB allocation) takes ~300ms, setting bit number 2 :sup:`30` -1
           (128MB allocation) takes ~80ms, setting bit number 2 :sup:`28` -1
           (32MB allocation) takes ~30ms and setting bit number 2 :sup:`26` -1
           (8MB allocation) takes ~8ms. Note that once this first allocation is
           done, subsequent calls to :meth:`~tredis.RedisClient.setbit` for the
           same key will not have the allocation overhead.

        .. versionadded:: 0.2.0

        .. note:: **Time complexity**: ``O(1)``

        :param key: The key to get the bit from
        :type key: :class:`str`, :class:`bytes`
        :param int offset: The bit offset to fetch the bit from
        :param int bit: The value (``0`` or ``1``) to set for the bit
        :rtype: int
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        if 0 < bit > 1:
            raise ValueError('bit must be 1 or 0, not {}'.format(bit))
        return self._execute([b'SETBIT', key, ascii(offset), ascii(bit)])
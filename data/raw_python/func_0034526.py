def setrange(self, key, offset, value):
        """Overwrites part of the string stored at key, starting at the
        specified offset, for the entire length of value. If the offset is
        larger than the current length of the string at key, the string is
        padded with zero-bytes to make offset fit. Non-existing keys are
        considered as empty strings, so this command will make sure it holds a
        string large enough to be able to set value at offset.

        .. note:: The maximum offset that you can set is 2 :sup:`29` -1
           (536870911), as Redis Strings are limited to 512 megabytes. If you
           need to grow beyond this size, you can use multiple keys.

        .. warning:: When setting the last possible byte and the string value
           stored at key does not yet hold a string value, or holds a small
           string value, Redis needs to allocate all intermediate memory which
           can block the server for some time. On a 2010 MacBook Pro, setting
           byte number 536870911 (512MB allocation) takes ~300ms, setting byte
           number 134217728 (128MB allocation) takes ~80ms, setting bit number
           33554432 (32MB allocation) takes ~30ms and setting bit number
           8388608 (8MB allocation) takes ~8ms. Note that once this first
           allocation is done, subsequent calls to
           :meth:`~tredis.RedisClient.setrange` for the same key will not have
           the allocation overhead.

        .. versionadded:: 0.2.0

        .. note:: **Time complexity**: ``O(1)``, not counting the time taken to
           copy the new string in place. Usually, this string is very small so
           the amortized complexity is ``O(1)``. Otherwise, complexity is
           ``O(M)`` with ``M`` being the length of the value argument.

        :param key: The key to get the bit from
        :type key: :class:`str`, :class:`bytes`
        :param value: The value to set
        :type value: :class:`str`, :class:`bytes`, :class:`int`
        :returns: The length of the string after it was modified by the command
        :rtype: int
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute([b'SETRANGE', key, ascii(offset), value])
def hincrby(self, key, field, increment):
        """
        Increments the number stored at `field` in the hash stored at `key`.

        If `key` does not exist, a new key holding a hash is created.  If
        `field` does not exist the value is set to ``0`` before the operation
        is performed.  The range of values supported is limited to 64-bit
        signed integers.

        :param key: The key of the hash
        :type key: :class:`str`, :class:`bytes`
        :param field: name of the field to increment
        :type key: :class:`str`, :class:`bytes`
        :param increment: amount to increment by
        :type increment: int

        :returns: the value at `field` after the increment occurs
        :rtype: int

        """
        return self._execute(
            [b'HINCRBY', key, field, increment], format_callback=int)
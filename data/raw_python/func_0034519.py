def incrbyfloat(self, key, increment):
        """Increment the string representing a floating point number stored at
        key by the specified increment. If the key does not exist, it is set to
        0 before performing the operation. An error is returned if one of the
        following conditions occur:

          - The key contains a value of the wrong type (not a string).
          - The current key content or the specified increment are not
            parsable as a double precision floating point number.

        If the command is successful the new incremented value is stored as the
        new value of the key (replacing the old one), and returned to the
        caller as a string.

        Both the value already contained in the string key and the increment
        argument can be optionally provided in exponential notation, however
        the value computed after the increment is stored consistently in the
        same format, that is, an integer number followed (if needed) by a dot,
        and a variable number of digits representing the decimal part of the
        number. Trailing zeroes are always removed.

        The precision of the output is fixed at 17 digits after the decimal
        point regardless of the actual internal precision of the computation.

        .. versionadded:: 0.2.0

        .. note:: **Time complexity**: ``O(1)``

        :param key: The key to increment
        :type key: :class:`str`, :class:`bytes`
        :param float increment: The amount to increment by
        :returns: The value of key after the increment
        :rtype: bytes
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute([b'INCRBYFLOAT', key, ascii(increment)])
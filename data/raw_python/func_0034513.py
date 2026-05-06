def bitop(self, operation, dest_key, *keys):
        """Perform a bitwise operation between multiple keys (containing
        string values) and store the result in the destination key.

        The values for operation can be one of:

            - ``b'AND'``
            - ``b'OR'``
            - ``b'XOR'``
            - ``b'NOT'``
            - :data:`tredis.BITOP_AND` or ``b'&'``
            - :data:`tredis.BITOP_OR` or ``b'|'``
            - :data:`tredis.BITOP_XOR` or ``b'^'``
            - :data:`tredis.BITOP_NOT` or ``b'~'``

        ``b'NOT'`` is special as it only takes an input key, because it
        performs inversion of bits so it only makes sense as an unary operator.

        The result of the operation is always stored at ``dest_key``.

        **Handling of strings with different lengths**

        When an operation is performed between strings having different
        lengths, all the strings shorter than the longest string in the set are
        treated as if they were zero-padded up to the length of the longest
        string.

        The same holds true for non-existent keys, that are considered as a
        stream of zero bytes up to the length of the longest string.

        .. versionadded:: 0.2.0

        .. note:: **Time complexity**: ``O(N)``

        :param bytes operation: The operation to perform
        :param dest_key: The key to store the bitwise operation results to
        :type dest_key: :class:`str`, :class:`bytes`
        :param keys: One or more keys as keyword parameters for the bitwise op
        :type keys: :class:`str`, :class:`bytes`
        :return: The size of the string stored in the destination key, that is
                 equal to the size of the longest input string.
        :rtype: int
        :raises: :exc:`~tredis.exceptions.RedisError`, :exc:`ValueError`

        """
        if (operation not in _BITOPTS.keys()
                and operation not in _BITOPTS.values()):
            raise ValueError('Invalid operation value: {}'.format(operation))
        elif operation in [b'~', b'NOT'] and len(keys) > 1:
            raise ValueError('NOT can only be used with 1 key')

        if operation in _BITOPTS.keys():
            operation = _BITOPTS[operation]

        return self._execute([b'BITOP', operation, dest_key] + list(keys))
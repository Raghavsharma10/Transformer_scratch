def lpushx(self, key, *values):
        """
        Insert values at the head of an existing list.

        :param key: The list's key
        :type key: :class:`str`, :class:`bytes`
        :param values: One or more positional arguments to insert at the
            beginning of the list.  Each value is inserted at the beginning
            of the list individually (see discussion below).
        :returns: the length of the list after push operations, zero if
            `key` does not refer to a list
        :rtype: int
        :raises: :exc:`~tredis.exceptions.TRedisException`

        This method inserts `values` at the head of the list stored at `key`,
        only if `key` already exists and holds a list. In contrary to
        :meth:`.lpush`, no operation will be performed when key does not yet
        exist.

        .. note::

           **Time complexity**: ``O(1)``

        """
        return self._execute([b'LPUSHX', key] + list(values))
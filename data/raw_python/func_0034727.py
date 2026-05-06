def rpushx(self, key, *values):
        """
        Insert values at the tail of an existing list.

        :param key: The list's key
        :type key: :class:`str`, :class:`bytes`
        :param values: One or more positional arguments to insert at the
            tail of the list.
        :returns: the length of the list after push operations or
            zero if `key` does not refer to a list
        :rtype: int
        :raises: :exc:`~tredis.exceptions.TRedisException`

        This method inserts value at the tail of the list stored at `key`,
        only if `key` already exists and holds a list. In contrary to
        method:`.rpush`, no operation will be performed when `key` does not
        yet exist.

        .. note::

           **Time complexity**: ``O(1)``

        """
        return self._execute([b'RPUSHX', key] + list(values))
def rename(self, key, new_key):
        """Renames ``key`` to ``new_key``. It returns an error when the source
        and destination names are the same, or when ``key`` does not exist.
        If ``new_key`` already exists it is overwritten, when this happens
        :meth:`~tredis.RedisClient.rename` executes an implicit
        :meth:`~tredis.RedisClient.delete` operation, so if the deleted key
        contains a very big value it may cause high latency even if
        :meth:`~tredis.RedisClient.rename` itself is usually a constant-time
        operation.

        .. note::

           **Time complexity**: ``O(1)``

        :param key: The key to rename
        :type key: :class:`str`, :class:`bytes`
        :param new_key: The key to rename it to
        :type new_key: :class:`str`, :class:`bytes`
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute([b'RENAME', key, new_key], b'OK')
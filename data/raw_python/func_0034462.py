def delete(self, *keys):
        """Removes the specified keys. A key is ignored if it does not exist.
        Returns :data:`True` if all keys are removed.

        .. note::

           **Time complexity**: ``O(N)`` where ``N`` is the number of keys that
           will be removed. When a key to remove holds a value other than a
           string, the individual complexity for this key is ``O(M)`` where
           ``M`` is the number of elements in the list, set, sorted set or
           hash. Removing a single key that holds a string value is ``O(1)``.

        :param keys: One or more keys to remove
        :type keys: :class:`str`, :class:`bytes`
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute([b'DEL'] + list(keys), len(keys))
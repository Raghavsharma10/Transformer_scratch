def hset(self, key, field, value):
        """Sets `field` in the hash stored at `key` to `value`.

        If `key` does not exist, a new key holding a hash is created. If
        `field` already exists in the hash, it is overwritten.

        .. note::

           **Time complexity**: always ``O(1)``

        :param key: The key of the hash
        :type key: :class:`str`, :class:`bytes`
        :param field: The field in the hash to set
        :type key: :class:`str`, :class:`bytes`
        :param value: The value to set the field to
        :returns: ``1`` if `field` is a new field in the hash and `value`
            was set; otherwise, ``0`` if `field` already exists in the hash
            and the value was updated
        :rtype: int

        """
        return self._execute([b'HSET', key, field, value])
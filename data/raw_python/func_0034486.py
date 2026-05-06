def hsetnx(self, key, field, value):
        """
        Sets `field` in the hash stored at `key` only if it does not exist.

        Sets `field` in the hash stored at `key` only if `field` does not
        yet exist.  If `key` does not exist, a new key holding a hash is
        created.  If `field` already exists, this operation has no effect.

        .. note::

           *Time complexity*: ``O(1)``

        :param key: The key of the hash
        :type key: :class:`str`, :class:`bytes`
        :param field: The field in the hash to set
        :type key: :class:`str`, :class:`bytes`
        :param value: The value to set the field to
        :returns: ``1`` if `field` is a new field in the hash and `value`
            was set.  ``0`` if `field` already exists in the hash and
            no operation was performed
        :rtype: int

        """
        return self._execute([b'HSETNX', key, field, value])
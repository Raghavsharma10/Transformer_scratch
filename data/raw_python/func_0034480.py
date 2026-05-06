def hgetall(self, key):
        """
        Returns all fields and values of the has stored at `key`.

        The underlying redis `HGETALL`_ command returns an array of
        pairs.  This method converts that to a Python :class:`dict`.
        It will return an empty :class:`dict` when the key is not
        found.

        .. note::

           **Time complexity**: ``O(N)`` where ``N`` is the size
           of the hash.

        :param key: The key of the hash
        :type key: :class:`str`, :class:`bytes`
        :returns: a :class:`dict` of key to value mappings for all
            fields in the hash

        .. _HGETALL: http://redis.io/commands/hgetall

        """

        def format_response(value):
            return dict(zip(value[::2], value[1::2]))

        return self._execute(
            [b'HGETALL', key], format_callback=format_response)
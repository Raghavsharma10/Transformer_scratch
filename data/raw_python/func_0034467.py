def pexpire(self, key, timeout):
        """This command works exactly like :meth:`~tredis.RedisClient.pexpire`
        but the time to live of the key is specified in milliseconds instead of
        seconds.

        .. note::

           **Time complexity**: ``O(1)``

        :param key: The key to set an expiration for
        :type key: :class:`str`, :class:`bytes`
        :param int timeout: The number of milliseconds to set the timeout to
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute(
            [b'PEXPIRE', key, ascii(timeout).encode('ascii')], 1)
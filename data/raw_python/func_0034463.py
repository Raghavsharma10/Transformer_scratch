def expire(self, key, timeout):
        """Set a timeout on key. After the timeout has expired, the key will
        automatically be deleted. A key with an associated timeout is often
        said to be volatile in Redis terminology.

        The timeout is cleared only when the key is removed using the
        :meth:`~tredis.RedisClient.delete` method or overwritten using the
        :meth:`~tredis.RedisClient.set` or :meth:`~tredis.RedisClient.getset`
        methods. This means that all the operations that conceptually alter the
        value stored at the key without replacing it with a new one will leave
        the timeout untouched. For instance, incrementing the value of a key
        with :meth:`~tredis.RedisClient.incr`, pushing a new value into a
        list with :meth:`~tredis.RedisClient.lpush`, or altering the field
        value of a hash with :meth:`~tredis.RedisClient.hset` are all
        operations that will leave the timeout untouched.

        The timeout can also be cleared, turning the key back into a persistent
        key, using the :meth:`~tredis.RedisClient.persist` method.

        If a key is renamed with :meth:`~tredis.RedisClient.rename`,
        the associated time to live is transferred to the new key name.

        If a key is overwritten by :meth:`~tredis.RedisClient.rename`, like in
        the case of an existing key ``Key_A`` that is overwritten by a call
        like ``client.rename(Key_B, Key_A)`` it does not matter if the original
        ``Key_A`` had a timeout associated or not, the new key ``Key_A`` will
        inherit all the characteristics of ``Key_B``.

        .. note::

           **Time complexity**: ``O(1)``

        :param key: The key to set an expiration for
        :type key: :class:`str`, :class:`bytes`
        :param int timeout: The number of seconds to set the timeout to
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute(
            [b'EXPIRE', key, ascii(timeout).encode('ascii')], 1)
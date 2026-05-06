def scan(self, cursor=0, pattern=None, count=None):
        """The :meth:`~tredis.RedisClient.scan` command and the closely related
        commands :meth:`~tredis.RedisClient.sscan`,
        :meth:`~tredis.RedisClient.hscan` and :meth:`~tredis.RedisClient.zscan`
        are used in order to incrementally iterate over a collection of
        elements.

        - :meth:`~tredis.RedisClient.scan` iterates the set of keys in the
          currently selected Redis database.
        - :meth:`~tredis.RedisClient.sscan` iterates elements of Sets types.
        - :meth:`~tredis.RedisClient.hscan` iterates fields of Hash types and
          their associated values.
        - :meth:`~tredis.RedisClient.zscan` iterates elements of Sorted Set
          types and their associated scores.

        **Basic usage**

        :meth:`~tredis.RedisClient.scan` is a cursor based iterator.
        This means that at every call of the command, the server returns an
        updated cursor that the user needs to use as the cursor argument in
        the next call.

        An iteration starts when the cursor is set to ``0``, and terminates
        when the cursor returned by the server is ``0``.

        For more information on :meth:`~tredis.RedisClient.scan`,
        visit the `Redis docs on scan <http://redis.io/commands/scan>`_.

        .. note::

           **Time complexity**: ``O(1)`` for every call. ``O(N)`` for a
           complete iteration, including enough command calls for the cursor to
           return back to ``0``. ``N`` is the number of elements inside the
           collection.

        :param int cursor: The server specified cursor value or ``0``
        :param pattern: An optional pattern to apply for key matching
        :type pattern: :class:`str`, :class:`bytes`
        :param int count: An optional amount of work to perform in the scan
        :rtype: int, list
        :returns: A tuple containing the cursor and the list of keys
        :raises: :exc:`~tredis.exceptions.RedisError`

        """

        def format_response(value):
            """Format the response from redis

            :param tuple value: The return response from redis
            :rtype: tuple(int, list)

            """
            return int(value[0]), value[1]

        command = [b'SCAN', ascii(cursor).encode('ascii')]
        if pattern:
            command += [b'MATCH', pattern]
        if count:
            command += [b'COUNT', ascii(count).encode('ascii')]
        return self._execute(command, format_callback=format_response)
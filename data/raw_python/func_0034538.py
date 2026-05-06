def zadd(self, key, *members, **kwargs):
        """Adds all the specified members with the specified scores to the
        sorted set stored at key. It is possible to specify multiple score /
        member pairs. If a specified member is already a member of the sorted
        set, the score is updated and the element reinserted at the right
        position to ensure the correct ordering.

        If key does not exist, a new sorted set with the specified members as
        sole members is created, like if the sorted set was empty. If the key
        exists but does not hold a sorted set, an error is returned.

        The score values should be the string representation of a double
        precision floating point number. +inf and -inf values are valid values
        as well.

        **Members parameters**

        ``members`` could be either:
        - a single dict where keys correspond to scores and values to elements
        - multiple strings paired as score then element

        .. code:: python

            yield client.zadd('myzset', {'1': 'one', '2': 'two'})
            yield client.zadd('myzset', '1', 'one', '2', 'two')

        **ZADD options (Redis 3.0.2 or greater)**

        ZADD supports a list of options. Options are:

        - ``xx``: Only update elements that already exist. Never add elements.
        - ``nx``: Don't update already existing elements. Always add new
            elements.
        - ``ch``: Modify the return value from the number of new elements
            added, to the total number of elements changed (CH is an
            abbreviation of changed). Changed elements are new elements added
            and elements already existing for which the score was updated. So
            elements specified in the command having the same score as they had
            in the past are not counted. Note: normally the return value of
            ``ZADD`` only counts the number of new elements added.
        - ``incr``: When this option is specified ``ZADD`` acts like
            :meth:`~tredis.RedisClient.zincrby`. Only one score-element pair
            can be specified in this mode.

        .. note::

           **Time complexity**: ``O(log(N))`` for each item added, where ``N``
           is the number of elements in the sorted set.

        :param key: The key of the sorted set
        :type key: :class:`str`, :class:`bytes`
        :param members: Elements to add
        :type members: :class:`dict`, :class:`str`, :class:`bytes`
        :keyword bool xx: Only update elements that already exist
        :keyword bool nx: Don't update already existing elements
        :keyword bool ch: Return the number of changed elements
        :keyword bool incr: Increment the score of an element
        :rtype: int, :class:`str`, :class:`bytes`
        :returns: Number of elements changed, or the new score if incr is set
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        xx = kwargs.pop('xx', False)
        nx = kwargs.pop('nx', False)
        ch = kwargs.pop('ch', False)
        incr = kwargs.pop('incr', False)
        command = [b'ZADD', key]
        if xx:
            command += ['XX']
        if nx:
            command += ['NX']
        if ch:
            command += ['CH']
        if incr:
            command += ['INCR']

        if len(members) == 1:
            for k in members[0]:
                command += [k, members[0][k]]
        else:
            command += list(members)
        return self._execute(command)
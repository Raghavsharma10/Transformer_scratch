def sort(self,
             key,
             by=None,
             external=None,
             offset=0,
             limit=None,
             order=None,
             alpha=False,
             store_as=None):
        """Returns or stores the elements contained in the list, set or sorted
        set at key. By default, sorting is numeric and elements are compared by
        their value interpreted as double precision floating point number.

        The ``external`` parameter is used to specify the
        `GET <http://redis.io/commands/sort#retrieving-external-keys>_`
        parameter for retrieving external keys. It can be a single string
        or a list of strings.

        .. note::

           **Time complexity**: ``O(N+M*log(M))`` where ``N`` is the number of
           elements in the list or set to sort, and ``M`` the number of
           returned elements. When the elements are not sorted, complexity is
           currently ``O(N)`` as there is a copy step that will be avoided in
           next releases.

        :param key: The key to get the refcount for
        :type key: :class:`str`, :class:`bytes`

        :param by: The optional pattern for external sorting keys
        :type by: :class:`str`, :class:`bytes`
        :param external: Pattern or list of patterns to return external keys
        :type external: :class:`str`, :class:`bytes`, list
        :param int offset: The starting offset when using limit
        :param int limit: The number of elements to return
        :param order: The sort order - one of ``ASC`` or ``DESC``
        :type order: :class:`str`, :class:`bytes`
        :param bool alpha: Sort the results lexicographically
        :param store_as: When specified, the key to store the results as
        :type store_as: :class:`str`, :class:`bytes`, None
        :rtype: list|int
        :raises: :exc:`~tredis.exceptions.RedisError`
        :raises: :exc:`ValueError`

        """
        if order and order not in [b'ASC', b'DESC', 'ASC', 'DESC']:
            raise ValueError('invalid sort order "{}"'.format(order))

        command = [b'SORT', key]
        if by:
            command += [b'BY', by]
        if external and isinstance(external, list):
            for entry in external:
                command += [b'GET', entry]
        elif external:
            command += [b'GET', external]
        if limit:
            command += [
                b'LIMIT',
                ascii(offset).encode('utf-8'),
                ascii(limit).encode('utf-8')
            ]
        if order:
            command.append(order)
        if alpha is True:
            command.append(b'ALPHA')
        if store_as:
            command += [b'STORE', store_as]

        return self._execute(command)
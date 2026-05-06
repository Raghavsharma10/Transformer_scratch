def smove(self, source, destination, member):
        """Move member from the set at source to the set at destination. This
        operation is atomic. In every given moment the element will appear to
        be a member of source or destination for other clients.

        If the source set does not exist or does not contain the specified
        element, no operation is performed and :data:`False` is returned.
        Otherwise, the element is removed from the source set and added to the
        destination set. When the specified element already exists in the
        destination set, it is only removed from the source set.

        An error is returned if source or destination does not hold a set
        value.

        .. note::

           **Time complexity**: ``O(1)``

        :param source: The source set key
        :type source: :class:`str`, :class:`bytes`
        :param destination: The destination set key
        :type destination: :class:`str`, :class:`bytes`
        :param member: The member value to move
        :type member: :class:`str`, :class:`bytes`
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute([b'SMOVE', source, destination, member], 1)
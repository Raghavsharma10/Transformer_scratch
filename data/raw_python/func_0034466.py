def move(self, key, db):
        """Move key from the currently selected database (see
        :meth:`~tredis.RedisClient.select`) to the specified destination
        database. When key already exists in the destination database, or it
        does not exist in the source database, it does nothing. It is possible
        to use :meth:`~tredis.RedisClient.move` as a locking primitive because
        of this.

        .. note::

           **Time complexity**: ``O(1)``

        :param key: The key to move
        :type key: :class:`str`, :class:`bytes`
        :param int db: The database number
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        return self._execute([b'MOVE', key, ascii(db).encode('ascii')], 1)
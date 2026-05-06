def migrate(self,
                host,
                port,
                key,
                destination_db,
                timeout,
                copy=False,
                replace=False):
        """Atomically transfer a key from a source Redis instance to a
        destination Redis instance. On success the key is deleted from the
        original instance and is guaranteed to exist in the target instance.

        The command is atomic and blocks the two instances for the time
        required to transfer the key, at any given time the key will appear to
        exist in a given instance or in the other instance, unless a timeout
        error occurs.

        .. note::

           **Time complexity**: This command actually executes a DUMP+DEL in
           the source instance, and a RESTORE in the target instance. See the
           pages of these commands for time complexity. Also an ``O(N)`` data
           transfer between the two instances is performed.

        :param host: The host to migrate the key to
        :type host: bytes, str
        :param int port: The port to connect on
        :param key: The key to migrate
        :type key: bytes, str
        :param int destination_db: The database number to select
        :param int timeout: The maximum idle time in milliseconds
        :param bool copy: Do not remove the key from the local instance
        :param bool replace: Replace existing key on the remote instance
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        command = [
            b'MIGRATE', host,
            ascii(port).encode('ascii'), key,
            ascii(destination_db).encode('ascii'),
            ascii(timeout).encode('ascii')
        ]
        if copy is True:
            command.append(b'COPY')
        if replace is True:
            command.append(b'REPLACE')
        return self._execute(command, b'OK')
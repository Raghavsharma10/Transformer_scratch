def wait(self, num_slaves, timeout=0):
        """his command blocks the current client until all the previous write
        commands are successfully transferred and acknowledged by at least the
        specified number of slaves. If the timeout, specified in milliseconds,
        is reached, the command returns even if the specified number of slaves
        were not yet reached.

        The command will always return the number of slaves that acknowledged
        the write commands sent before the :meth:`~tredis.RedisClient.wait`
        command, both in the case where the specified number of slaves are
        reached, or when the timeout is reached.

        .. note::

           **Time complexity**: ``O(1)``

        :param int num_slaves: Number of slaves to acknowledge previous writes
        :param int timeout: Timeout in milliseconds
        :rtype: int
        :raises: :exc:`~tredis.exceptions.RedisError`

        """
        command = [
            b'WAIT',
            ascii(num_slaves).encode('ascii'),
            ascii(timeout).encode('ascii')
        ]
        return self._execute(command)
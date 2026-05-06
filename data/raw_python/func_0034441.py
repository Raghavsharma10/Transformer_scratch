def select(self, index=0):
        """Select the DB with having the specified zero-based numeric index.
        New connections always use DB ``0``.

        :param int index: The database to select
        :rtype: bool
        :raises: :exc:`~tredis.exceptions.RedisError`
        :raises: :exc:`~tredis.exceptions.InvalidClusterCommand`

        """
        if self._clustering:
            raise exceptions.InvalidClusterCommand
        future = self._execute(
            [b'SELECT', ascii(index).encode('ascii')], b'OK')

        def on_selected(f):
            self._connection.database = index

        self.io_loop.add_future(future, on_selected)
        return future
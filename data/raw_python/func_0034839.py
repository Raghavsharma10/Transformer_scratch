def _execute(self, parts, expectation=None, format_callback=None):
        """Really execute a redis command

        :param list parts: The list of command parts
        :param mixed expectation: Optional response expectation

        :rtype: :class:`~tornado.concurrent.Future`
        :raises: :exc:`~tredis.exceptions.SubscribedError`

        """
        future = concurrent.TracebackFuture()

        try:
            command = self._build_command(parts)
        except ValueError as error:
            future.set_exception(error)
            return future

        def on_locked(_):
            if self.ready:
                if self._clustering:
                    cmd = Command(command, self._pick_cluster_host(parts),
                                  expectation, format_callback)
                else:
                    LOGGER.debug('Connection: %r', self._connection)
                    cmd = Command(command, self._connection, expectation,
                                  format_callback)
                LOGGER.debug('_execute(%r, %r, %r) on %s', cmd.command,
                             expectation, format_callback, cmd.connection.name)
                cmd.connection.execute(cmd, future)
            else:
                LOGGER.critical('Lock released & not ready, aborting command')

        # Wait until the cluster is ready, letting cluster discovery through
        if not self.ready and not self._connected.is_set():
            self.io_loop.add_future(
                self._connected.wait(),
                lambda f: self.io_loop.add_future(self._busy.acquire(), on_locked)
            )
        else:
            self.io_loop.add_future(self._busy.acquire(), on_locked)

        # Release the lock when the future is complete
        self.io_loop.add_future(future, lambda r: self._busy.release())
        return future
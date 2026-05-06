def _on_read_only_error(self, command, future):
        """Invoked when a Redis node returns an error indicating it's in
        read-only mode. It will use the ``INFO REPLICATION`` command to
        attempt to find the master server and failover to that, reissuing
        the command to that server.

        :param command: The command that was being executed
        :type command: tredis.client.Command
        :param future: The execution future
        :type future: tornado.concurrent.Future

        """
        failover_future = concurrent.TracebackFuture()

        def on_replication_info(_):
            common.maybe_raise_exception(failover_future)
            LOGGER.debug('Failover closing current read-only connection')
            self._closing = True
            database = self._connection.database
            self._connection.close()
            self._connected.clear()
            self._connect_future = concurrent.Future()

            info = failover_future.result()
            LOGGER.debug('Failover connecting to %s:%s', info['master_host'],
                         info['master_port'])
            self._connection = _Connection(
                info['master_host'], info['master_port'], database, self._read,
                self._on_closed, self.io_loop, self._clustering)

            # When the connection is re-established, re-run the command
            self.io_loop.add_future(
                self._connect_future,
                lambda f: self._connection.execute(
                    command._replace(connection=self._connection), future))

            # Use the normal connection processing flow when connecting
            self.io_loop.add_future(self._connection.connect(),
                                    self._on_connected)

        if self._clustering:
            command.connection.set_readonly(True)

        LOGGER.debug('%s is read-only, need to failover to new master',
                     command.connection.name)

        cmd = Command(
            self._build_command(['INFO', 'REPLICATION']), self._connection,
            None, common.format_info_response)

        self.io_loop.add_future(failover_future, on_replication_info)
        cmd.connection.execute(cmd, failover_future)
def _on_connected(self, future):
        """Invoked when connections have been established. If the client is
        in clustering mode, it will kick of the discovery step if needed. If
        not, it will select the configured database.

        :param future: The connection future
        :type future: tornado.concurrent.Future

        """
        if future.exception():
            self._connect_future.set_exception(future.exception())
            return

        conn = future.result()
        LOGGER.debug('Connected to %s (%r, %r, %r)', conn.name,
                     self._clustering, self._discovery, self._connected)
        if self._clustering:
            self._cluster[conn.name] = conn
            if not self._discovery:
                self.io_loop.add_future(self.cluster_nodes(),
                                        self._on_cluster_discovery)
            elif self.ready:
                LOGGER.debug('Cluster nodes all connected')
                if not self._connect_future.done():
                    self._connect_future.set_result(True)
                self._connected.set()
        else:

            def on_selected(sfuture):
                LOGGER.debug('Initial setup and selection processed')
                if sfuture.exception():
                    self._connect_future.set_exception(sfuture.exception())
                else:
                    self._connect_future.set_result(True)
                self._connected.set()

            select_future = concurrent.Future()
            self.io_loop.add_future(select_future, on_selected)
            self._connection = conn
            cmd = Command(
                self._build_command(['SELECT', str(conn.database)]),
                self._connection, None, None)
            cmd.connection.execute(cmd, select_future)
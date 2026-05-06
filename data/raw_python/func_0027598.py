def _establish_connection(self, conn: BrokerConnection) -> None:
        """
        We don't use a pool here. We only have one consumer connection per process, so
        we get no value from a pool, and we want to use a heartbeat to keep the consumer
        collection alive, which does not work with a pool
        :return: the connection to the transport
        """
        try:
            self._logger.debug("Establishing connection.")
            self._conn = conn.ensure_connection(max_retries=3)
            self._logger.debug('Got connection: %s', conn.as_uri())
        except kombu_exceptions.OperationalError as oe:
            self._logger.error("Error connecting to RMQ, could not retry %s", oe)
            # Try to clean up the mess
            if self._conn is not None:
                self._conn.close()
            else:
                conn.close()
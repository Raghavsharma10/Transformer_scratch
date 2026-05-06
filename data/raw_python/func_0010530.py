def connection(self) -> Iterator[amqp.Connection]:
        """Returns a new connection as a context manager."""
        TCP_USER_TIMEOUT = 18  # constant is available on Python 3.6+.
        socket_settings = {TCP_USER_TIMEOUT: self.config.TCP_USER_TIMEOUT}

        if sys.platform.startswith('darwin'):
            del socket_settings[TCP_USER_TIMEOUT]

        conn = amqp.Connection(
            host="%s:%s" % (self.config.RABBIT_HOST, self.config.RABBIT_PORT),
            userid=self.config.RABBIT_USER,
            password=self.config.RABBIT_PASSWORD,
            virtual_host=self.config.RABBIT_VIRTUAL_HOST,
            connect_timeout=self.config.RABBIT_CONNECT_TIMEOUT,
            read_timeout=self.config.RABBIT_READ_TIMEOUT,
            write_timeout=self.config.RABBIT_WRITE_TIMEOUT,
            socket_settings=socket_settings,
            heartbeat=self.config.RABBIT_HEARTBEAT,
        )
        conn.connect()
        logger.info('Connected to RabbitMQ')
        with _safe_close(conn):
            yield conn
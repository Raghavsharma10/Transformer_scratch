def establish_connection(self):
        """Establish connection to the AMQP broker."""
        conninfo = self.connection
        if not conninfo.port:
            conninfo.port = self.default_port
        credentials = pika.PlainCredentials(conninfo.userid,
                                            conninfo.password)
        return self._connection_cls(pika.ConnectionParameters(
                                           conninfo.hostname,
                                           port=conninfo.port,
                                           virtual_host=conninfo.virtual_host,
                                           credentials=credentials))
def establish_connection(self):
        """Establish connection to the AMQP broker."""
        conninfo = self.connection
        if not conninfo.hostname:
            raise KeyError("Missing hostname for AMQP connection.")
        if conninfo.userid is None:
            raise KeyError("Missing user id for AMQP connection.")
        if conninfo.password is None:
            raise KeyError("Missing password for AMQP connection.")
        if not conninfo.port:
            conninfo.port = self.default_port
        conn = amqp.Connection(host=conninfo.hostname,
                               port=conninfo.port,
                               userid=conninfo.userid,
                               password=conninfo.password,
                               virtual_host=conninfo.virtual_host)
        return conn
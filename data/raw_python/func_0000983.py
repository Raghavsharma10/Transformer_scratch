def connect(self):
        """Opens a HTTP connection to the RPC server."""
        logger.debug("Opening connection to %s:%s", self.url.hostname, self.url.port)
        try:
            self.connection = httplib.HTTPConnection(self.url.hostname, self.url.port)
            self.connection.connect()
        except (httplib.HTTPException, socket.error) as e:
            raise errors.InterfaceError('Unable to connect to the specified service', e)
def close(self):
        """Closes the HTTP connection to the RPC server."""
        if self.connection is not None:
            logger.debug("Closing connection to %s:%s", self.url.hostname, self.url.port)
            try:
                self.connection.close()
            except httplib.HTTPException:
                logger.warning("Error while closing connection", exc_info=True)
            self.connection = None
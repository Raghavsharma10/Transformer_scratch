def connection(self):
        """Return an SqlAlchemy connection."""
        if not self._connection:
            logger.debug('Opening connection to: {}'.format(self.dsn))
            self._connection = self.engine.connect()
            logger.debug('Opened connection to: {}'.format(self.dsn))

        # logger.debug("Opening connection to: {}".format(self.dsn))
        return self._connection
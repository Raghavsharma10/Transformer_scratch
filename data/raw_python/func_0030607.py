def close(self):
        """ Closes connection to database. """
        if getattr(self, '_connection', None):
            logger.debug('Closing postgresql connection.')
            self._connection.close()
            self._connection = None
        if getattr(self, '_engine', None):
            self._engine.dispose()
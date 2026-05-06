def close(self):
        """ Closes connection to sqlite database. """
        if getattr(self, '_connection', None):
            logger.debug('Closing sqlite connection.')
            self._connection.close()
            self._connection = None
def _connect(self):
        """Try to create a connection to the database if not yet connected.
        """
        if self._connection is not None:
            raise RuntimeError('Close connection first.')
        self._connection = connect(self._database, **self._kwds)
        self._connection.isolation_level = None
def _connect(self):
        """Establish connection to MySQL Database."""
        if self._connParams:
            self._conn = MySQLdb.connect(**self._connParams)
        else:
            self._conn = MySQLdb.connect('')
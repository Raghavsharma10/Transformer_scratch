def db(self, connection_string=None):
        """Gets the SQLALchemy session for this request"""

        connection_string = connection_string or self.settings["db"]

        if not hasattr(self, "_db_conns"):
            self._db_conns = {}
        if not connection_string in self._db_conns:
            self._db_conns[connection_string] = oz.sqlalchemy.session(connection_string=connection_string)

        return self._db_conns[connection_string]
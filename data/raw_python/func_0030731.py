def exists(self):
        """Return True if the database exists, or for Sqlite, which will create the file on the
        first reference, the file has been initialized with the root config """

        if self.driver == 'sqlite' and not os.path.exists(self.path):
            return False

        # init engine
        self.engine

        try:
            # Since we are using the connection, rather than the session, need to
            # explicitly set the search path.

            from sqlalchemy.engine.reflection import Inspector

            inspector = Inspector.from_engine(self.engine)

            if 'config' in inspector.get_table_names(schema=self._schema):
                return True
            else:
                return False
        finally:
            self.close_connection()
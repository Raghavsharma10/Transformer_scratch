def migrate(self) -> None:
        """ Migrate

        Perform a database migration, upgrading to the latest schema level.
        """

        assert self.ormSessionCreator, "ormSessionCreator is not defined"

        connection = self._dbEngine.connect()
        isDbInitialised = self._dbEngine.dialect.has_table(
            connection, 'alembic_version',
            schema=self._metadata.schema)
        connection.close()

        if isDbInitialised or not self._enableCreateAll:
            self._doMigration(self._dbEngine)

        else:
            self._doCreateAll(self._dbEngine)

        if self._enableForeignKeys:
            self.checkForeignKeys(self._dbEngine)
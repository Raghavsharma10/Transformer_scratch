def _migrateStorageSchema(self, metadata: MetaData) -> None:
        """ Initialise the DB

        This method is called by the platform between the load() and start() calls.
        There should be no need for a plugin to call this method it's self.

        :param metadata: the SQLAlchemy metadata for this plugins schema

        """

        relDir = self._packageCfg.config.storage.alembicDir(require_string)
        alembicDir = os.path.join(self.rootDir, relDir)
        if not os.path.isdir(alembicDir): raise NotADirectoryError(alembicDir)

        self._dbConn = DbConnection(
            dbConnectString=self.platform.dbConnectString,
            metadata=metadata,
            alembicDir=alembicDir,
            enableCreateAll=False
        )

        self._dbConn.migrate()
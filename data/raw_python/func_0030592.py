def database_dsn(self):
        """Substitute the root dir into the database DSN, for Sqlite"""

        if not self._config.library.database:
            return 'sqlite:///{root}/library.db'.format(root=self._root)

        return self._config.library.database.format(root=self._root)
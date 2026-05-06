def create_savepoint(self, savepoint):
        """Creates a new savepoint.

        :param savepoint: the name of the savepoint to create
        :raise: pydbal.exception.DBALConnectionError
        """
        if not self._platform.is_savepoints_supported():
            raise DBALConnectionError.savepoints_not_supported()
        self.ensure_connected()
        self._platform.create_savepoint(savepoint)
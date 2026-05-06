def rollback_savepoint(self, savepoint):
        """Rolls back to the given savepoint.

        :param savepoint: the name of the savepoint to rollback to
        :raise: pydbal.exception.DBALConnectionError
        """
        if not self._platform.is_savepoints_supported():
            raise DBALConnectionError.savepoints_not_supported()
        self.ensure_connected()
        self._platform.rollback_savepoint(savepoint)
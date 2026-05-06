def release_savepoint(self, savepoint):
        """Releases the given savepoint.

        :param savepoint: the name of the savepoint to release
        :raise: pydbal.exception.DBALConnectionError
        """
        if not self._platform.is_savepoints_supported():
            raise DBALConnectionError.savepoints_not_supported()
        if self._platform.is_release_savepoints_supported():
            self.ensure_connected()
            self._platform.release_savepoint(savepoint)
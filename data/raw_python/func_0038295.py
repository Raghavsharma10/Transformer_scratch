def execute(self, command):
        """Execute a single SQL query without returning a result."""
        self._cursor.execute(command)
        self._commit()
        return True
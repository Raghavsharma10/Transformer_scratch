def executemany(self, command, params=None, max_attempts=5):
        """Execute multiple SQL queries without returning a result."""
        attempts = 0
        while attempts < max_attempts:
            try:
                # Execute statement
                self._cursor.executemany(command, params)
                self._commit()
                return True
            except Exception as e:
                attempts += 1
                self.reconnect()
                continue
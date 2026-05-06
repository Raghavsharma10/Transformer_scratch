def close(self):
        """
        Close the database connection and render self invalid. Any subsequent
        re-use of self will raise an error.
        """
        self._cursor.close()
        self._db.close()
        self._cursor = self._db = self._cache = None
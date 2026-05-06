def list(self):
        """List the tables in the database"""
        connection = self._backend._get_connection()
        return list(self._backend.list(connection))
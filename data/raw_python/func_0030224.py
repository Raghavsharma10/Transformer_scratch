def clean(self):
        """Remove all of the tables and data from the warehouse"""
        connection = self._backend._get_connection()
        self._backend.clean(connection)
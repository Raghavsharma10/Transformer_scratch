def reset(self):
        """ Drops index table. """
        query = """
            DROP TABLE identifier_index;
        """
        self.backend.library.database.connection.execute(query)
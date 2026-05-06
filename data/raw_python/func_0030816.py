def _delete(self, vid=None):
        """ Deletes given dataset from index.

        Args:
            vid (str): dataset vid.

        """
        query = text("""
            DELETE FROM dataset_index
            WHERE vid = :vid;
        """)
        self.backend.library.database.connection.execute(query, vid=vid)
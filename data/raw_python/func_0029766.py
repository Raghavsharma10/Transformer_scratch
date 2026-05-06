def _delete(self, vid=None):
        """ Deletes given dataset from index.

        Args:
            vid (str): dataset vid.

        """
        assert vid is not None
        query = text("""
            DELETE FROM dataset_index
            WHERE vid = :vid;
        """)
        self.execute(query, vid=vid)
def is_indexed(self, dataset):
        """ Returns True if dataset is already indexed. Otherwise returns False. """
        query = text("""
            SELECT vid
            FROM dataset_index
            WHERE vid = :vid;
        """)
        result = self.backend.library.database.connection.execute(query, vid=dataset.vid)
        return bool(result.fetchall())
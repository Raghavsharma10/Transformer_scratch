def is_indexed(self, partition):
        """ Returns True if partition is already indexed. Otherwise returns False. """
        query = text("""
            SELECT vid
            FROM partition_index
            WHERE vid = :vid;
        """)
        result = self.execute(query, vid=partition.vid)
        return bool(result.fetchall())
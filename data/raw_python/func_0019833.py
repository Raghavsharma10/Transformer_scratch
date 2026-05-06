def getBgWriterStats(self):
        """Returns Global Background Writer and Checkpoint Activity stats.
        
        @return: Nested dictionary of stats.
        
        """
        info_dict = {}
        if self.checkVersion('8.3'):
            cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT * FROM pg_stat_bgwriter")
            info_dict = cur.fetchone()
        return info_dict
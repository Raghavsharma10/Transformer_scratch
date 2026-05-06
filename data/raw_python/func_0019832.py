def getLockStatsDB(self):
        """Returns the number of active lock discriminated by database.
        
        @return: : Dictionary of stats.
        
        """
        info_dict = {'all': {},
                     'wait': {}}
        cur = self._conn.cursor()
        cur.execute("SELECT d.datname, l.granted, COUNT(*) FROM pg_database d "
                    "JOIN pg_locks l ON d.oid=l.database "
                    "GROUP BY d.datname, l.granted;")
        rows = cur.fetchall()
        for (db, granted, cnt) in rows:
            info_dict['all'][db] = info_dict['all'].get(db, 0) + cnt
            if not granted:
                info_dict['wait'][db] = info_dict['wait'].get(db, 0) + cnt
        return info_dict
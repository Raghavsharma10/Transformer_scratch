def getProcessDatabase(self):
        """Returns number of processes discriminated by database name.
        
        @return: Dictionary mapping database name to number of processes.
        
        """
        info_dict = {}
        cur = self._conn.cursor()
        cur.execute("""SHOW FULL PROCESSLIST;""")
        rows = cur.fetchall()
        if rows:
            for row in rows:
                db = row[3]
                info_dict[db] = info_dict.get(db, 0) + 1
        return info_dict
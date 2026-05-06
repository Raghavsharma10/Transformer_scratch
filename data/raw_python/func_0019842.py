def getDatabases(self):
        """Returns list of databases.
        
        @return: List of databases.
        
        """
        cur = self._conn.cursor()
        cur.execute("""SHOW DATABASES;""")
        rows = cur.fetchall()
        if rows:
            return [row[0] for row in rows]
        else:
            return []
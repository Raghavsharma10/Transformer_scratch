def getStorageEngines(self):
        """Returns list of supported storage engines.
        
        @return: List of storage engine names.
        
        """
        cur = self._conn.cursor()
        cur.execute("""SHOW STORAGE ENGINES;""")
        rows = cur.fetchall()
        if rows:
            return [row[0].lower() for row in rows if row[1] in ['YES', 'DEFAULT']]
        else:
            return []
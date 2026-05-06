def getParams(self):
        """Returns dictionary of all run-time parameters.
        
        @return: Dictionary of all Run-time parameters.
        
        """
        cur = self._conn.cursor()
        cur.execute("SHOW GLOBAL VARIABLES")
        rows = cur.fetchall()
        info_dict = {}
        for row in rows:
            key = row[0]
            val = util.parse_value(row[1])
            info_dict[key] = val
        return info_dict
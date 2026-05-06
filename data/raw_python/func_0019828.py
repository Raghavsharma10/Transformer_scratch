def getParam(self, key):
        """Returns value of Run-time Database Parameter 'key'.
        
        @param key: Run-time parameter name.
        @return:    Run-time parameter value.
        
        """
        cur = self._conn.cursor()
        cur.execute("SHOW %s" % key)
        row = cur.fetchone()
        return util.parse_value(row[0])
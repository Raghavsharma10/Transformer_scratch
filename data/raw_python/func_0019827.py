def _simpleQuery(self, query):
        """Executes simple query which returns a single column.
        
        @param query: Query string.
        @return:      Query result string.
        
        """
        cur = self._conn.cursor()
        cur.execute(query)
        row = cur.fetchone()
        return util.parse_value(row[0])
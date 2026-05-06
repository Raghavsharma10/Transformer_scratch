def _createStatsDict(self, headers, rows):
        """Utility method that returns database stats as a nested dictionary.
        
        @param headers: List of columns in query result.
        @param rows:    List of rows in query result.
        @return:        Nested dictionary of values.
                        First key is the database name and the second key is the 
                        statistics counter name. 
            
        """
        dbstats = {}
        for row in rows:
            dbstats[row[0]] = dict(zip(headers[1:], row[1:]))
        return dbstats
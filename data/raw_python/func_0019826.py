def _createTotalsDict(self, headers, rows):
        """Utility method that returns totals for database statistics.
        
        @param headers: List of columns in query result.
        @param rows:    List of rows in query result.
        @return:        Dictionary of totals for each statistics column. 
            
        """
        totals = [sum(col) for col in zip(*rows)[1:]]
        return dict(zip(headers[1:], totals))
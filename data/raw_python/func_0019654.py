def unregisterFilter(self, column):
        """Unregister filter on a column of the table.
        
        @param column: The column header.
        
        """
        if self._filters.has_key(column):
            del self._filters[column]
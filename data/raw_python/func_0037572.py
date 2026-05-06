def count_rows_duplicates(self, table, cols='*'):
        """Get the number of rows that do not contain distinct values."""
        return self.count_rows(table, '*') - self.count_rows_distinct(table, cols)
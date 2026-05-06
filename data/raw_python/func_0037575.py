def get_unique_column(self, table):
        """Determine if any of the columns in a table contain exclusively unique values."""
        for col in self.get_columns(table):
            if self.count_rows_duplicates(table, col) == 0:
                return col
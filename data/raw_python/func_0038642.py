def select_all(self, table, limit=MAX_ROWS_PER_QUERY, execute=True):
        """Query all rows and columns from a table."""
        # Determine if a row per query limit should be set
        num_rows = self.count_rows(table)
        if num_rows > limit:
            return self._select_batched(table, '*', num_rows, limit, execute=execute)
        else:
            return self.select(table, '*', execute=execute)
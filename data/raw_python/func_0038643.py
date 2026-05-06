def select_distinct(self, table, cols='*', execute=True):
        """Query distinct values from a table."""
        return self.select(table, cols, execute, select_type='SELECT DISTINCT')
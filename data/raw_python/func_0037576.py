def get_duplicate_vals(self, table, column):
        """Retrieve duplicate values in a column of a table."""
        query = 'SELECT {0} FROM {1} GROUP BY {0} HAVING COUNT(*) > 1'.format(join_cols(column), wrap(table))
        return self.fetch(query)
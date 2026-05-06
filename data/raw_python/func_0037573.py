def count_rows(self, table, cols='*'):
        """Get the number of rows in a particular table."""
        query = 'SELECT COUNT({0}) FROM {1}'.format(join_cols(cols), wrap(table))
        result = self.fetch(query)
        return result if result is not None else 0
def count_rows_distinct(self, table, cols='*'):
        """Get the number distinct of rows in a particular table."""
        return self.fetch('SELECT COUNT(DISTINCT {0}) FROM {1}'.format(join_cols(cols), wrap(table)))
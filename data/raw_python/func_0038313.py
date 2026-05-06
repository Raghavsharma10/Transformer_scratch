def drop_index(self, table, column):
        """Drop an index from a table."""
        self.execute('ALTER TABLE {0} DROP INDEX {1}'.format(wrap(table), column))
        self._printer('\tDropped index from column {0}'.format(column))
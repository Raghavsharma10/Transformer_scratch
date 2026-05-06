def drop_column(self, table, name):
        """Remove a column to an existing table."""
        try:
            self.execute('ALTER TABLE {0} DROP COLUMN {1}'.format(wrap(table), name))
            self._printer('\tDropped column {0} from {1}'.format(name, table))
        except ProgrammingError:
            self._printer("\tCan't DROP '{0}'; check that column/key exists in '{1}'".format(name, table))
        return name
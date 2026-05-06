def add_comment(self, table, column, comment):
        """Add a comment to an existing column in a table."""
        col_def = self.get_column_definition(table, column)
        query = "ALTER TABLE {0} MODIFY COLUMN {1} {2} COMMENT '{3}'".format(table, column, col_def, comment)
        self.execute(query)
        self._printer('\tAdded comment to column {0}'.format(column))
        return True
def set_foreign_key(self, parent_table, parent_column, child_table, child_column):
        """Create a Foreign Key constraint on a column from a table."""
        self.execute('ALTER TABLE {0} ADD FOREIGN KEY ({1}) REFERENCES {2}({3})'.format(parent_table, parent_column,
                                                                                        child_table, child_column))
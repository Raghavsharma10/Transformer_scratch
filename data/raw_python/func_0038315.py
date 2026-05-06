def add_auto_increment(self, table, name):
        """Modify an existing column."""
        # Get current column definition and add auto_incrementing
        definition = self.get_column_definition(table, name) + ' AUTO_INCREMENT'

        # Concatenate and execute modify statement
        self.execute("ALTER TABLE {0} MODIFY {1}".format(table, definition))
        self._printer('\tAdded AUTO_INCREMENT to column {0}'.format(name))
        return True
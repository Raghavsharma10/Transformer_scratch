def get_column_definition(self, table, column):
        """Retrieve the column definition statement for a column from a table."""
        # Parse column definitions for match
        for col in self.get_column_definition_all(table):
            if col.strip('`').startswith(column):
                return col.strip(',')
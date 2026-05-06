def get_column_definition_all(self, table):
        """Retrieve the column definition statement for all columns in a table."""
        # Get complete table definition
        col_defs = self.get_table_definition(table).split('\n')

        # Return only column definitions
        return [i[0:-1].strip().replace(',', ', ') for i in col_defs if i.strip().startswith('`')]
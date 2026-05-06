def get_database_rows(self, tables=None, database=None):
        """Retrieve a dictionary of table keys and list of rows values for every table."""
        # Get table data and columns from source database
        source = database if database else self.database
        tables = tables if tables else self.tables

        # Get database select queries
        commands = self._get_select_commands(source, tables)

        # Execute select commands
        return self._execute_select_commands(source, commands)
def _copy_database_data_clientside(self, tables, source, destination):
        """Copy the data from a table into another table."""
        # Retrieve database rows
        rows = self.get_database_rows(tables, source)

        # Retrieve database columns
        cols = self.get_database_columns(tables, source)

        # Validate rows and columns
        for r in list(rows.keys()):
            assert r in tables
        for c in list(cols.keys()):
            assert c in tables

        # Change database to destination
        self.change_db(destination)

        # Get insert queries
        insert_queries = self._get_insert_commands(rows, cols)

        # Execute insert queries
        self._execute_insert_commands(insert_queries)
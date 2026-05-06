def drop(self, table):
        """
        Drop a table from a database.

        Accepts either a string representing a table name or a list of strings
        representing a table names.
        """
        existing_tables = self.tables
        if isinstance(table, (list, set, tuple)):
            for t in table:
                self._drop(t, existing_tables)
        else:
            self._drop(table, existing_tables)
        return table
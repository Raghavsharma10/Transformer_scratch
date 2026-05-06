def set_primary_keys_auto(self, tables=None):
        """
        Create primary keys for every table in the connected database.

        Checks that each table has a primary key.  If a table does not have a key
        then each column is analyzed to determine if it contains only unique values.
        If no columns exist containing only unique values then a new 'ID' column
        is created to serve as a auto_incrementing primary key.
        """
        # Retrieve list of tables if not provided
        tables = tables if tables else self.tables

        # Resolve primary keys and return list of table, primary_key tuples
        return [(table, self.set_primary_key_auto(table)) for table in tables]
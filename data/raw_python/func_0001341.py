def _valid_table_name(self, table_name):
        """Check if the table name is obviously invalid.
        """
        if table_name is None or not len(table_name.strip()):
            raise ValueError("Invalid table name: %r" % table_name)
        return table_name.strip()
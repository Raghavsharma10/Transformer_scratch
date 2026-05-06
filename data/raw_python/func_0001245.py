def _valid_table_name(self, table):
        """Check if the table name is obviously invalid.
        """
        if table is None or not len(table.strip()):
            raise ValueError("Invalid table name: %r" % table)
        return table.strip()
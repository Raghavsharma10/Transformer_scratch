def column_types(self):
        """Return a dict mapping column name to type for all columns in table
        """
        column_types = {}
        for c in self.sqla_columns:
            column_types[c.name] = c.type
        return column_types
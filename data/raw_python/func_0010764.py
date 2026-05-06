def _query_helper(self, by=None):
        """
        Internal helper for preparing queries.
        """
        if by is None:
            primary_keys = self.table.primary_key.columns.keys()

            if len(primary_keys) > 1:
                warnings.warn("WARNING: MORE THAN 1 PRIMARY KEY FOR TABLE %s. "
                              "USING THE FIRST KEY %s." %
                              (self.table.name, primary_keys[0]))

            if not primary_keys:
                raise NoPrimaryKeyException("Table %s needs a primary key for"
                                            "the .last() method to work properly. "
                                            "Alternatively, specify an ORDER BY "
                                            "column with the by= argument. " %
                                            self.table.name)
            id_col = primary_keys[0]
        else:
            id_col = by

        if self.column is None:
            col = "*"
        else:
            col = self.column.name

        return col, id_col
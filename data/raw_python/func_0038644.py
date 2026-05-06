def select(self, table, cols, execute=True, select_type='SELECT', return_type=list):
        """Query every row and only certain columns from a table."""
        # Validate query type
        select_type = select_type.upper()
        assert select_type in SELECT_QUERY_TYPES

        # Concatenate statement
        statement = '{0} {1} FROM {2}'.format(select_type, join_cols(cols), wrap(table))
        if not execute:
            # Return command
            return statement

        # Retrieve values
        values = self.fetch(statement)
        return self._return_rows(table, cols, values, return_type)
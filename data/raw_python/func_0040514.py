def _produce_return(self, cursor):
        """ Return the one result.
        """
        results = cursor.fetchmany(2)
        if len(results) != 1:
            return None

        # Return the one row, or the one column.
        row = results[0]
        if self._row_formatter is not None:
            row = self._row_formatter(row, cursor)
        elif len(row) == 1:
            row = row[0]

        return row
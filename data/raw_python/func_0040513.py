def _produce_return(self, cursor):
        """ Get the rows from the cursor and apply the row formatter.

        :return: sequence of rows, or a generator if a row formatter has to be
            applied
        """
        results = cursor.fetchall()

        # Format rows within a generator?
        if self._row_formatter is not None:
            return (self._row_formatter(r, cursor) for r in results)

        return results
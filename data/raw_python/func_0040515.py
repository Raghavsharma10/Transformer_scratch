def _row_generator(self, cursor):
        """ Yields individual rows until no more rows
        exist in query result. Applies row formatter if such exists.
        """
        rowset = cursor.fetchmany(self._arraysize)
        while rowset:
            if self._row_formatter is not None:
                rowset = (self._row_formatter(r, cursor) for r in rowset)
            for row in rowset:
                yield row
            rowset = cursor.fetchmany(self._arraysize)
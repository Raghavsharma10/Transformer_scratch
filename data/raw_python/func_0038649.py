def select_where_like(self, table, cols, where_col, start=None, end=None, anywhere=None,
                          index=(None, None), length=None):
        """
        Query rows from a table where a specific pattern is found in a column.

        MySQL syntax assumptions:
            (%) The percent sign represents zero, one, or multiple characters.
            (_) The underscore represents a single character.

        :param table: Name of the table
        :param cols: List, tuple or set of columns or string with single column name
        :param where_col: Column to check pattern against
        :param start: Value to be found at the start
        :param end: Value to be found at the end
        :param anywhere: Value to be found anywhere
        :param index: Value to be found at a certain index
        :param length: Minimum character length
        :return: Queried rows
        """
        # Retrieve search pattern
        pattern = self._like_pattern(start, end, anywhere, index, length)

        # Concatenate full statement and execute
        statement = "SELECT {0} FROM {1} WHERE {2} LIKE '{3}'".format(join_cols(cols), wrap(table), where_col, pattern)
        return self.fetch(statement)
def select_where_between(self, table, cols, where_col, between):
        """
        Query rows from a table where a columns value is found between two values.

        :param table: Name of the table
        :param cols: List, tuple or set of columns or string with single column name
        :param where_col: Column to check values against
        :param between: Tuple with min and max values for comparison
        :return: Queried rows
        """
        # Unpack WHERE clause dictionary into tuple
        min_val, max_val = between

        # Concatenate full statement and execute
        statement = "SELECT {0} FROM {1} WHERE {2} BETWEEN {3} AND {4}".format(join_cols(cols), wrap(table), where_col,
                                                                               min_val, max_val)
        return self.fetch(statement)
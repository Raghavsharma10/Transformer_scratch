def select_where(self, table, cols, where, return_type=list):
        """
        Query certain rows from a table where a particular value is found.

        cols parameter can be passed as a iterable (list, set, tuple) or a string if
        only querying a single column.  where parameter can be passed as a two or three
        part tuple.  If only two parts are passed the assumed operator is equals(=).

        :param table: Name of table
        :param cols: List, tuple or set of columns or string with single column name
        :param where: WHERE clause, accepts either a two or three part tuple
            two-part: (where_column, where_value)
            three-part: (where_column, comparison_operator, where_value)
        :param return_type: Type, type to return values in
        :return: Queried rows
        """
        # Unpack WHERE clause dictionary into tuple
        if isinstance(where, (list, set)):
            # Multiple WHERE clause's (separate with AND)
            clauses = [self._where_clause(clause) for clause in where]
            where_statement = ' AND '.join(clauses)
        else:
            where_statement = self._where_clause(where)

        # Concatenate full statement and execute
        statement = "SELECT {0} FROM {1} WHERE {2}".format(join_cols(cols), wrap(table), where_statement)
        values = self.fetch(statement)
        return self._return_rows(table, cols, values, return_type)
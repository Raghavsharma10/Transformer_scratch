def update(self, table, values, identifier):
        """Updates a table row with specified data by given identifier.

        :param table: the expression of the table to update quoted or unquoted
        :param values: a dictionary containing column-value pairs
        :param identifier: the update criteria; a dictionary containing column-value pairs
        :return: the number of affected rows
        :rtype: int
        """
        assert isinstance(values, dict)
        assert isinstance(identifier, dict)

        sb = self.sql_builder().update(table)
        for column, value in values.iteritems():
            sb.set(column, sb.create_positional_parameter(value))
        for column, value in identifier.iteritems():
            func = self._expr.in_ if isinstance(value, (list, tuple)) else self._expr.eq
            sb.and_where(func(column, sb.create_positional_parameter(value)))
        return sb.execute()
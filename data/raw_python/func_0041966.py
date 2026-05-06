def delete(self, table, identifier):
        """Deletes a table row by given identifier.

        :param table: the expression of the table to update quoted or unquoted
        :param identifier: the delete criteria; a dictionary containing column-value pairs
        :return: the number of affected rows
        :rtype: int
        """
        assert isinstance(identifier, dict)

        sb = self.sql_builder().delete(table)
        for column, value in identifier.iteritems():
            func = self._expr.in_ if isinstance(value, (list, tuple)) else self._expr.eq
            sb.and_where(func(column, sb.create_positional_parameter(value)))
        return sb.execute()
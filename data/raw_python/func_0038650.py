def _where_clause(where):
        """
        Unpack a where clause tuple and concatenate a MySQL WHERE statement.

        :param where: 2 or 3 part tuple containing a where_column and a where_value (optional operator)
        :return: WHERE clause statement
        """
        assert isinstance(where, tuple)
        if len(where) == 3:
            where_col, operator, where_val = where
        else:
            where_col, where_val = where
            operator = '='
        assert operator in SELECT_WHERE_OPERATORS

        # Concatenate WHERE clause (ex: **first_name='John'**)
        return "{0}{1}'{2}'".format(where_col, operator, where_val)
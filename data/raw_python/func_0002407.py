def _get_aliased_pivot_columns(self):
        """
        Get the pivot columns for the relation.

        :rtype: list
        """
        defaults = [self._foreign_key, self._other_key]

        columns = []

        for column in defaults + self._pivot_columns:
            value = '%s.%s AS pivot_%s' % (self._table, column, column)
            if value not in columns:
                columns.append('%s.%s AS pivot_%s' % (self._table, column, column))

        return columns
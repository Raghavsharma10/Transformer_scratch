def _get_select_columns(self, columns=None):
        """
        Set the select clause for the relation query.

        :param columns: The columns
        :type columns: list

        :rtype: list
        """
        if columns == ['*'] or columns is None:
            columns = ['%s.*' % self._related.get_table()]

        return columns + self._get_aliased_pivot_columns()
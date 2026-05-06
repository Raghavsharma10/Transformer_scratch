def increment(self, column, amount=1, extras=None):
        """
        Increment a column's value by a given amount

        :param column: The column to increment
        :type column: str

        :param amount: The amount by which to increment
        :type amount: int

        :param extras: Extra columns
        :type extras: dict

        :return: The number of rows affected
        :rtype: int
        """
        wrapped = self._grammar.wrap(column)

        if extras is None:
            extras = {}

        columns = {
            column: self.raw('%s + %s' % (wrapped, amount))
        }
        columns.update(extras)

        return self.update(**columns)
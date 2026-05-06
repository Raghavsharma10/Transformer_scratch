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
        extras = self._add_updated_at_column(extras)

        return self._query.increment(column, amount, extras)
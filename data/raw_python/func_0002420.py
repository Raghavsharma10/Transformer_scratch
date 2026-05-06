def or_having(self, column, operator=None, value=None):
        """
        Add a "having" clause to the query

        :param column: The column
        :type column: str

        :param operator: The having clause operator
        :type operator: str

        :param value: The having clause value
        :type value: mixed

        :return: The current QueryBuilder instance
        :rtype: QueryBuilder
        """
        return self.having(column, operator, value, 'or')
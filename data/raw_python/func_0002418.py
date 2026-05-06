def join_where(self, table, one, operator, two, type='inner'):
        """
        Add a "join where" clause to the query

        :param table: The table to join with, can also be a JoinClause instance
        :type table: str or JoinClause

        :param one: The first column of the join condition
        :type one: str

        :param operator: The operator of the join condition
        :type operator: str

        :param two: The second column of the join condition
        :type two: str

        :param type: The join type
        :type type: str

        :return: The current QueryBuilder instance
        :rtype: QueryBuilder
        """
        return self.join(table, one, operator, two, type, True)
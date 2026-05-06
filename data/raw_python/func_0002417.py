def join(self, table, one=None,
             operator=None, two=None, type='inner', where=False):
        """
        Add a join clause to the query

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

        :param where: Whether to use a "where" rather than a "on"
        :type where: bool

        :return: The current QueryBuilder instance
        :rtype: QueryBuilder
        """
        if isinstance(table, JoinClause):
            self.joins.append(table)
        else:
            if one is None:
                raise ArgumentError('Missing "one" argument')

            join = JoinClause(table, type)

            self.joins.append(join.on(
                one, operator, two, 'and', where
            ))

        return self
def where_has(self, relation, extra, operator='>=', count=1):
        """
        Add a relationship count condition to the query with where clauses.

        :param relation: The relation to count
        :type relation: str

        :param extra: The extra query
        :type extra: Builder or callable

        :param operator: The operator
        :type operator: str

        :param count: The count
        :type count: int

        :rtype: Builder
        """
        return self.has(relation, operator, count, 'and', extra)
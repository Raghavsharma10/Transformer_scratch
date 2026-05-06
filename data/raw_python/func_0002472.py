def or_has(self, relation, operator='>=', count=1):
        """
        Add a relationship count condition to the query with an "or".

        :param relation: The relation to count
        :type relation: str

        :param operator: The operator
        :type operator: str

        :param count: The count
        :type count: int

        :rtype: Builder
        """
        return self.has(relation, operator, count, 'or')
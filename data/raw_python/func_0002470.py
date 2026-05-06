def doesnt_have(self, relation, boolean='and', extra=None):
        """
        Add a relationship count to the query.

        :param relation: The relation to count
        :type relation: str

        :param boolean: The boolean value
        :type boolean: str

        :param extra: The extra query
        :type extra: Builder or callable

        :rtype: Builder
        """
        return self.has(relation, '<', 1, boolean, extra)
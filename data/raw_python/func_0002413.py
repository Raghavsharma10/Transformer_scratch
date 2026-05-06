def query(self):
        """
        Begin a fluent query

        :return: A QueryBuilder instance
        :rtype: QueryBuilder
        """
        query = QueryBuilder(self, self._query_grammar, self._post_processor)

        return query
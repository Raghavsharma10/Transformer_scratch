def union(self, query, all=False):
        """
        Add a union statement to the query

        :param query: A QueryBuilder instance
        :type query: QueryBuilder

        :param all: Whether it is a "union all" statement
        :type all: bool

        :return: The query
        :rtype: QueryBuilder
        """
        self.unions.append({
            'query': query,
            'all': all
        })

        return self.merge_bindings(query)
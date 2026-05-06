def _where_in_sub(self, column, query, boolean, negate=False):
        """
        Add a where in with a sub select to the query

        :param column: The column
        :type column: str

        :param query: A QueryBuilder instance
        :type query: QueryBuilder

        :param boolean: The boolean operator
        :type boolean: str

        :param negate: Whether it is a not where in
        :param negate: bool

        :return: The current QueryBuilder instance
        :rtype: QueryBuilder
        """
        if negate:
            type = 'not_in_sub'
        else:
            type = 'in_sub'

        self.wheres.append({
            'type': type,
            'column': column,
            'query': query,
            'boolean': boolean
        })

        self.merge_bindings(query)

        return self
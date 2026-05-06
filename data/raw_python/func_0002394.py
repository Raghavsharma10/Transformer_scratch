def compile_update(self, query, values):
        """
        Compile an update statement into SQL

        :param query: A QueryBuilder instance
        :type query: QueryBuilder

        :param values: The update values
        :type values: dict

        :return: The compiled update
        :rtype: str
        """
        sql = super(MySqlQueryGrammar, self).compile_update(query, values)

        if query.orders:
            sql += ' %s' % self._compile_orders(query, query.orders)

        if query.limit_:
            sql += ' %s' % self._compile_limit(query, query.limit_)

        return sql.rstrip()
def compile_select(self, query):
        """
        Compile a select query into SQL

        :param query: A QueryBuilder instance
        :type query: QueryBuilder

        :return: The compiled sql
        :rtype: str
        """
        sql = super(MySqlQueryGrammar, self).compile_select(query)

        if query.unions:
            sql = '(%s) %s' % (sql, self._compile_unions(query))

        return sql
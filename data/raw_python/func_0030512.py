def query(self, connection, query, fetch=True):
        """ Creates virtual tables for all partitions found in the query and executes query.

        Args:
            query (str): sql query
            fetch (bool): fetch result from database if True, do not fetch overwise.

        """

        self.install_module(connection)

        statements = sqlparse.parse(sqlparse.format(query, strip_comments=True))

        # install all partitions and replace table names in the query.
        #
        logger.debug('Finding and installing all partitions from query. \n    query: {}'.format(query))
        new_query = []

        if len(statements) > 1:
            raise BadSQLError("Can only query a single statement")

        if len(statements) == 0:
            raise BadSQLError("DIdn't get any statements in '{}'".format(query))

        statement = statements[0]

        logger.debug( 'Searching statement for partition ref.\n    statement: {}'.format(statement.to_unicode()))

        #statement = self.install_statement(connection, statement.to_unicode())

        logger.debug(
            'Executing updated query after partition install.'
            '\n    query before update: {}\n    query to execute (updated query): {}'
            .format(statement, new_query))

        return self._execute(connection, statement.to_unicode(), fetch=fetch)
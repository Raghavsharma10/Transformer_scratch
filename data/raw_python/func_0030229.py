def parse_sql(self, asql):
        """ Executes all sql statements from asql.

        Args:
            library (library.Library):
            asql (str): ambry sql query - see https://github.com/CivicKnowledge/ambry/issues/140 for details.
        """
        import sqlparse

        statements = sqlparse.parse(sqlparse.format(asql, strip_comments=True))
        parsed_statements = []
        for statement in statements:

            statement_str = statement.to_unicode().strip()

            for preprocessor in self._backend.sql_processors():
                statement_str = preprocessor(statement_str, self._library, self._backend, self.connection)

            parsed_statements.append(statement_str)

        return parsed_statements
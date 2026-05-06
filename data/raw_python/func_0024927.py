def execute(self, statement, *args, **kwargs):
        """
        This convenience method will execute the query passed in as is.  For
        more complex functionality you may want to use the sqlalchemy engine
        directly, but this serves as an example implementation.

        :param select_query: SQL statement to execute that will identify the
            resultset of interest.

        """
        with self.engine.connect() as conn:
            s = sqlalchemy.sql.text(statement)
            return conn.execute(s, **kwargs)
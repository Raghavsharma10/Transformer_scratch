def execute(self, stmt, **params):
        """Execute a SQL statement.

        The statement may be a string SQL string,
        an :func:`sqlalchemy.sql.expression.select` construct, or a 
        :func:`sqlalchemy.sql.expression.text` 
        construct.

        """
        return self.session.execute(sql.text(stmt, bind=self.bind), **params)
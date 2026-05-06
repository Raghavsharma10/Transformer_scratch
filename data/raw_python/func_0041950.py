def query(self, sql, *args, **kwargs):
        """Executes an SQL SELECT query, returning a result set as a Statement object.

        :param sql: query to execute
        :param args: parameters iterable
        :param kwargs: parameters iterable
        :return: result set as a Statement object
        :rtype: pydbal.statement.Statement
        """
        self.ensure_connected()
        stmt = Statement(self)
        stmt.execute(sql, *args, **kwargs)
        return stmt
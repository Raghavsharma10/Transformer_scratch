def execute(self, sql, *args, **kwargs):
        """Executes an SQL INSERT/UPDATE/DELETE query with the given parameters and returns the number of affected rows.

        :param sql: statement to execute
        :param args: parameters iterable
        :param kwargs: parameters iterable
        :return: number of affected rows
        :rtype: int
        """
        self.ensure_connected()
        return Statement(self).execute(sql, *args, **kwargs)
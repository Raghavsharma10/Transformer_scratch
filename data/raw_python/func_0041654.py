def execute(self, sql, *args, **kwargs):
        """Executes an SQL INSERT/UPDATE/DELETE query with the given parameters and returns the number of affected rows.

        :param sql: statement to execute
        :param args: parameters iterable
        :param kwargs: parameters iterable
        :return: number of affected rows
        :rtype: int
        """
        with self.locked() as conn:
            return conn.execute(sql, *args, **kwargs)
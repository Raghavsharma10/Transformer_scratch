def fetch(self, sql, *args, **kwargs):
        """Executes an SQL SELECT query and returns the first row or `None`.

        :param sql: statement to execute
        :param args: parameters iterable
        :param kwargs: parameters iterable
        :return: the first row or `None`
        """
        with self.locked() as conn:
            return conn.query(sql, *args, **kwargs).fetch()
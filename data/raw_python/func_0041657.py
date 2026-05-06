def fetch_column(self, sql, *args, **kwargs):
        """Executes an SQL SELECT query and returns the first column of the first row or `None`.

        :param sql: statement to execute
        :param args: parameters iterable
        :param kwargs: parameters iterable
        :return: the first row of the first column or `None`
        """
        with self.locked() as conn:
            return conn.query(sql, *args, **kwargs).fetch_column()
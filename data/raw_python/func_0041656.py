def fetch_all(self, sql, *args, **kwargs):
        """Executes an SQL SELECT query and returns all selected rows.

        :param sql: statement to execute
        :param args: parameters iterable
        :param kwargs: parameters iterable
        :return: all selected rows
        :rtype: list
        """
        with self.locked() as conn:
            return conn.query(sql, *args, **kwargs).fetch_all()
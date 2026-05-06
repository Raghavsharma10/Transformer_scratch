def query(self, sql, *args, **kwargs):
        """Executes an SQL SELECT query and returns rows generator.

        :param sql: query to execute
        :param args: parameters iterable
        :param kwargs: parameters iterable
        :return: rows generator
        :rtype: generator
        """
        with self.locked() as conn:
            for row in conn.query(sql, *args, **kwargs):
                yield row
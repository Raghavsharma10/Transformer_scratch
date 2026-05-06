def _executing(self, sql, params=[]):
        """
        Execute and yield rows in a way to support :meth:`close_connection`.
        """
        with self.connection() as connection:
            for row in connection.execute(sql, params):
                yield row
                if not self._db:
                    return
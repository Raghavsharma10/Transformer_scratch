def execute(self, sql, params=None):
        """Just a pointer to engine.execute
        """
        # wrap in a transaction to ensure things are committed
        # https://github.com/smnorris/pgdata/issues/3
        with self.engine.begin() as conn:
            result = conn.execute(sql, params)
        return result
def compile(self):
        """
        Compile SQL and return 3-tuple ``(sql, params, keys)``.

        Example usage::

            (sql, params, keys) = sc.compile()
            for row in cursor.execute(sql, params):
                record = dict(zip(keys, row))

        """
        params = self.column_params + self.join_params + self.params
        if self.limit and self.limit >= 0:
            self.sql_limit = 'LIMIT ?'
            params += [self.limit]
        return (self.sql, params, self.keys)
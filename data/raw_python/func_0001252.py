def query_one(self, sql, params=None):
        """Grab just one record
        """
        r = self.engine.execute(sql, params)
        return r.fetchone()
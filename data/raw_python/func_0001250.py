def mogrify(self, sql, params):
        """Return the query string with parameters added
        """
        conn = self.engine.raw_connection()
        cursor = conn.cursor()
        return cursor.mogrify(sql, params)
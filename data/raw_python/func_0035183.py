def execute_sql(self, query):
        """
        Executes a given query string on an open postgres database.

        """
        c = self.con.cursor()
        c.execute(query)
        result = []
        if c.rowcount > 0:
            try:
                result = c.fetchall()
            except psycopg2.ProgrammingError:
                pass
        return result
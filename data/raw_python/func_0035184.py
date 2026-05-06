def _database_exists(self):
        """
        Check if the database exists.

        """
        con = psycopg2.connect(host=self.host, database="postgres",
            user=self.user, password=self.password, port=self.port)
        query_check = "select datname from pg_catalog.pg_database"
        query_check += " where datname = '{0}';".format(self.dbname)
        c = con.cursor()
        c.execute(query_check)
        result = c.fetchall()
        if len(result) > 0:
            return True
        return False
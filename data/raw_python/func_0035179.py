def reset_database(self):
        """
        Re-create an empty database.

        """
        if self._database_exists():
            con = psycopg2.connect(host=self.host, database="postgres",
                user=self.user, password=self.password, port=self.port)
            con.set_isolation_level(
                psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            query = "DROP DATABASE {0};".format(self.dbname)
            c = con.cursor()
            c.execute(query)
            con.close()
        self.create_database()
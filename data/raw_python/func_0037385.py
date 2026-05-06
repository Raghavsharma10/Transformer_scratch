def create(self, sql=None):
        """CREATE this DATABASE.

        @param sql: (Optional) A string of psql (such as might be generated
        by pg_dump); it will be executed by psql(1) after creating the
        database.
        @type sql: str

        @rtype: None
        """
        create_sql = 'CREATE DATABASE {self.db_name} WITH OWNER {self.user}'
        create_sql = create_sql.format(**vars())
        self.super_psql(['-c', create_sql])
        if sql:
            self.psql_string(sql)
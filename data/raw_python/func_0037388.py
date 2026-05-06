def sql(self, input_string, *args):
        """Execute a SQL command using the Python DBI directly.

        Connection parameters are taken from self.  Autocommit is in effect.

        Example: .sql('SELECT %s FROM %s WHERE age > %s', 'name', 'table1',
            '45')

        @param input_string: A string of SQL.  May contain %s or %(name)s
        format specifiers; they are replaced with corresponding values taken
        from args.

        @param args: zero or more parameters to interpolate into the string.
        Note that they're passed individually, not as a single tuple.

        @return: Whatever .fetchall() returns.
        """
        """
        # I advise against using sqlalchemy here (it's more complicated than
        # what we need), but here's an implementation Just In Case.  -jps
        import psycopg2, sqlalchemy
        engine = sqlalchemy.create_engine(
            'postgres://%s@%s:%s/%s' %
            (self.user, self.host, self.port, self.db_name),
            echo=False, poolclass=sqlalchemy.pool.NullPool)
        connection = engine.connect()
        result = connection.execute(input_string, *args)
        try:
            # sqlalchemy 0.6.7 offers a result.returns_rows attribute, but
            # no prior version offers anything comparable.  A tacky
            # workaround...
            try:
                return result.fetchall()
            except psycopg2.ProgrammingError:
                return None
        finally:
            result.close()
            connection.close()
        """
        psycopg2 = importlib.import_module('psycopg2')
        importlib.import_module('psycopg2.extensions')
        connection = psycopg2.connect(
            user=self.user, host=self.host,
            port=self.port, database=self.db_name)
        connection.set_isolation_level(
            psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        try:
            cursor = connection.cursor()
            cursor.execute(input_string, args)
            # No way to ask whether any rows were returned, so just try it...
            try:
                return cursor.fetchall()
            except psycopg2.ProgrammingError:
                return None
        finally:
            connection.close()
def _execute(self, connection, query, fetch=True):
        """ Executes given query and returns result.

        Args:
            connection: connection to postgres database who stores mpr data.
            query (str): sql query
            fetch (boolean, optional): if True, fetch query result and return it. If False, do not fetch.

        Returns:
            iterable with query result or None if fetch is False.

        """
        # execute query
        with connection.cursor() as cursor:
            cursor.execute(query)
            if fetch:
                return cursor.fetchall()
            else:
                cursor.execute('COMMIT;')
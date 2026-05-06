def _execute(self, connection, query, fetch=True):
        """ Executes given query using given connection.

        Args:
            connection (apsw.Connection): connection to the sqlite db who stores mpr data.
            query (str): sql query
            fetch (boolean, optional): if True, fetch query result and return it. If False, do not fetch.

        Returns:
            iterable with query result.

        """
        cursor = connection.cursor()

        try:
            cursor.execute(query)
        except Exception as e:
            from ambry.mprlib.exceptions import BadSQLError
            raise BadSQLError("Failed to execute query: {}; {}".format(query, e))

        if fetch:
            return cursor.fetchall()
        else:
            return cursor
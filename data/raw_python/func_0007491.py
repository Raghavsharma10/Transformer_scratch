def fetchall(self, query, *args):
        """
        Returns all results of the given query.

        :param query: The query to be executed as a `str`.
        :param params: A `tuple` of parameters that will be replaced for
                       placeholders in the query.
        :return: A `list` of `tuple`s with each field being one element in the
                 `tuple`.
        """
        cursor = self.connection.cursor()

        try:
            cursor.execute(query, args)

            return cursor.fetchall()
        finally:
            cursor.close()
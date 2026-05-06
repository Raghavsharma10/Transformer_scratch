def fetchone(self, query, *args):
        """
        Returns the first result of the given query.

        :param query: The query to be executed as a `str`.
        :param params: A `tuple` of parameters that will be replaced for
                       placeholders in the query.
        :return: The retrieved row with each field being one element in a
                 `tuple`.
        """
        cursor = self.connection.cursor()

        try:
            cursor.execute(query, args)

            return cursor.fetchone()
        finally:
            cursor.close()
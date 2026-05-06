def execute(self, query, *params):
        """
        Executes a query and returns the identifier of the modified row.

        :param query: The query to be executed as a `str`.
        :param params: A `tuple` of parameters that will be replaced for
                       placeholders in the query.
        :return: A `long` identifying the last altered row.
        """
        cursor = self.connection.cursor()

        try:
            cursor.execute(query, params)

            self.connection.commit()

            return cursor.lastrowid
        finally:
            cursor.close()
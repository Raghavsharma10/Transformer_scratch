def update(self, table, values, identifier):
        """Updates a table row with specified data by given identifier.

        :param table: the expression of the table to update quoted or unquoted
        :param values: a dictionary containing column-value pairs
        :param identifier: the update criteria; a dictionary containing column-value pairs
        :return: the number of affected rows
        :rtype: int
        """
        with self.locked() as conn:
            return conn.update(table, values, identifier)
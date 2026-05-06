def delete(self, table, identifier):
        """Deletes a table row by given identifier.

        :param table: the expression of the table to update quoted or unquoted
        :param identifier: the delete criteria; a dictionary containing column-value pairs
        :return: the number of affected rows
        :rtype: int
        """
        with self.locked() as conn:
            return conn.delete(table, identifier)
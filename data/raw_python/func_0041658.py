def insert(self, table, values):
        """Inserts a table row with specified data.

        :param table: the expression of the table to insert data into, quoted or unquoted
        :param values: a dictionary containing column-value pairs
        :return: last inserted ID
        """
        with self.locked() as conn:
            return conn.insert(table, values)
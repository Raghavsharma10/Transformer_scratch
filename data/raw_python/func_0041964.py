def insert(self, table, values):
        """Inserts a table row with specified data.

        :param table: the expression of the table to insert data into, quoted or unquoted
        :param values: a dictionary containing column-value pairs
        :return: last inserted ID
        """
        assert isinstance(values, dict)

        sb = self.sql_builder().insert(table)
        for column, value in values.iteritems():
            values[column] = sb.create_positional_parameter(value)
        return sb.values(values).execute()
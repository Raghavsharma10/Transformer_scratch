def compile_insert(self, query, values):
        """
        Compile an insert SQL statement

        :param query: A QueryBuilder instance
        :type query: QueryBuilder

        :param values: The values to insert
        :type values: dict or list

        :return: The compiled statement
        :rtype: str
        """
        # Essentially we will force every insert to be treated as a batch insert which
        # simply makes creating the SQL easier for us since we can utilize the same
        # basic routine regardless of an amount of records given to us to insert.
        table = self.wrap_table(query.from__)

        if not isinstance(values, list):
            values = [values]

        columns = self.columnize(values[0].keys())

        # We need to build a list of parameter place-holders of values that are bound
        # to the query. Each insert should have the exact same amount of parameter
        # bindings so we can just go off the first list of values in this array.
        parameters = self.parameterize(values[0].values())

        value = ['(%s)' % parameters] * len(values)

        parameters = ', '.join(value)

        return 'INSERT INTO %s (%s) VALUES %s' % (table, columns, parameters)
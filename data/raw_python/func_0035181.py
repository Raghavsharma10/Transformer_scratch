def delete_index(self, cardinality):
        """
        Delete index for the table with the given cardinality.

        Parameters
        ----------
        cardinality : int
            The cardinality of the index to delete.

        """
        DatabaseConnector.delete_index(self, cardinality)

        query = "DROP INDEX IF EXISTS idx_{0}_gram_varchar;".format(cardinality)
        self.execute_sql(query)
        query = "DROP INDEX IF EXISTS idx_{0}_gram_normalized_varchar;".format(
            cardinality)
        self.execute_sql(query)
        query = "DROP INDEX IF EXISTS idx_{0}_gram_lower_varchar;".format(
            cardinality)
        self.execute_sql(query)
        query = "DROP INDEX IF EXISTS idx_{0}_gram_lower_normalized_varchar;".\
            format(cardinality)
        self.execute_sql(query)
        for i in reversed(range(cardinality)):
            if i != 0:
                query = "DROP INDEX IF EXISTS idx_{0}_gram_{1}_lower;".format(
                    cardinality, i)
                self.execute_sql(query)
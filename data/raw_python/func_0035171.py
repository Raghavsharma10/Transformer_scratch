def create_index(self, cardinality):
        """
        Create an index for the table with the given cardinality.

        Parameters
        ----------
        cardinality : int
            The cardinality to create a index for.

        """
        for i in reversed(range(cardinality)):
            if i != 0:
                query = "CREATE INDEX idx_{0}_gram_{1} ON _{0}_gram(word_{1});".format(cardinality, i)
                self.execute_sql(query)
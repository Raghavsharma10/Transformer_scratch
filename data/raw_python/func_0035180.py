def create_index(self, cardinality):
        """
        Create an index for the table with the given cardinality.

        Parameters
        ----------
        cardinality : int
            The cardinality to create a index for.

        """
        DatabaseConnector.create_index(self, cardinality)
        query = "CREATE INDEX idx_{0}_gram_varchar ON _{0}_gram(word varchar_pattern_ops);".format(cardinality)
        self.execute_sql(query)

        if self.lowercase:

            for i in reversed(range(cardinality)):
                if i != 0:
                    query = "CREATE INDEX idx_{0}_gram_{1}_lower ON _{0}_gram(LOWER(word_{1}));".format(cardinality, i)
                    self.execute_sql(query)

            if self.normalize:

                query = "CREATE INDEX idx_{0}_gram_lower_normalized_varchar ON _{0}_gram(NORMALIZE(LOWER(word)) varchar_pattern_ops);".format(cardinality)
                self.execute_sql(query)

            else:

                query = "CREATE INDEX idx_{0}_gram_lower_varchar ON _{0}_gram(LOWER(word) varchar_pattern_ops);".format(cardinality)
                self.execute_sql(query)

        elif self.normalize:

            query = "CREATE INDEX idx_{0}_gram_normalized_varchar ON _{0}_gram(NORMALIZE(word) varchar_pattern_ops);".format(cardinality)
            self.execute_sql(query)
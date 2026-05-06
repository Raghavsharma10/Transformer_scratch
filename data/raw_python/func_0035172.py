def ngrams(self, with_counts=False):
        """
        Returns all ngrams that are in the table.

        Parameters
        ----------
        None

        Returns
        -------
        ngrams : generator
            A generator for ngram tuples.

        """
        query = "SELECT "
        for i in reversed(range(self.cardinality)):
            if i != 0:
                query += "word_{0}, ".format(i)
            elif i == 0:
                query += "word"

        if with_counts:
            query += ", count"

        query += " FROM _{0}_gram;".format(self.cardinality)

        result = self.execute_sql(query)
        for row in result:
            yield tuple(row)
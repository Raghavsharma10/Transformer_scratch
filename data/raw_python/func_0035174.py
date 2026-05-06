def insert_ngram(self, ngram, count):
        """
        Inserts a given n-gram with count into the database.

        Parameters
        ----------
        ngram : iterable of str
            A list, set or tuple of strings.
        count : int
            The count for the given n-gram.

        """
        query = "INSERT INTO _{0}_gram {1};".format(len(ngram),
            self._build_values_clause(ngram, count))
        self.execute_sql(query)
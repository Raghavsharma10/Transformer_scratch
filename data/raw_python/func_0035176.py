def remove_ngram(self, ngram):
        """
        Removes a given ngram from the databae. The ngram has to be in the
        database, otherwise this method will stop with an error.

        Parameters
        ----------
        ngram : iterable of str
            A list, set or tuple of strings.

        """
        query = "DELETE FROM _{0}_gram".format(len(ngram))
        query += self._build_where_clause(ngram)
        query += ";"
        self.execute_sql(query)
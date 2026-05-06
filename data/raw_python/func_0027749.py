def _selectStuff(self, verb='SELECT'):
        """
        Return a generator which yields the massaged results of this query with
        a particular SQL verb.

        For an attribute query, massaged results are of the type of that
        attribute.  For an item query, they are items of the type the query is
        supposed to return.

        @param verb: a str containing the SQL verb to execute.  This really
        must be some variant of 'SELECT', the only two currently implemented
        being 'SELECT' and 'SELECT DISTINCT'.
        """
        sqlResults = self._runQuery(verb, self._queryTarget)
        for row in sqlResults:
            yield self._massageData(row)
def count(self):
        """Counts the number of results that could be accessed with the current parameters.

        :return:  number of objects matches to the query
        :rtype: int
        """
        # Save the existing rows and start parameters to see how many results were actually expected
        _rows = self._solr_params.get('rows', None)
        _start = self._solr_params.get('start', 0)
        if not self._solr_cache:
            # Get the count for everything
            self.set_params(rows=0)
            self._exec_query()
        number = self._solr_cache.get('num_found', -1)

        # If start value is bigger or equal than solr results' count, should return 0
        # If there are 10 results, but if start value is specified as 30, should return 0.
        if _start >= number:
            return 0
        else:
            # If 'start' is specified, then this many results from the start will not be accessible.
            number -= _start
            # If 'rows' is NOT specified, then all results are accessible
            # (minus the ones skipped with 'start')
            if _rows is None:
                return number
            # If 'rows' is specified, then this many results at most will be accessible. If we have
            # more than this many results found, then we can say that this many results are
            # accessible. If there are less results found than rows, then we can't give more
            # than found results.
            return number if number < _rows else _rows
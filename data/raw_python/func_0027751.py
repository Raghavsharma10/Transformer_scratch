def paginate(self, pagesize=20):
        """
        Split up the work of gathering a result set into multiple smaller
        'pages', allowing very large queries to be iterated without blocking
        for long periods of time.

        While simply iterating C{paginate()} is very similar to iterating a
        query directly, using this method allows the work to obtain the results
        to be performed on demand, over a series of different transaction.

        @param pagesize: the number of results gather in each chunk of work.
        (This is mostly for testing paginate's implementation.)
        @type pagesize: L{int}

        @return: an iterable which yields all the results of this query.
        """

        sort = self.sort
        oc = list(sort.orderColumns())
        if not oc:
            # You can't have an unsorted pagination.
            sort = self.tableClass.storeID.ascending
            oc = list(sort.orderColumns())
        if len(oc) != 1:
            raise RuntimeError("%d-column sorts not supported yet with paginate" %(len(oc),))
        sortColumn = oc[0][0]
        if oc[0][1] == 'ASC':
            sortOp = operator.gt
        else:
            sortOp = operator.lt
        if _isColumnUnique(sortColumn):
            # This is the easy case.  There is never a tie to be broken, so we
            # can just remember our last value and yield from there.  Right now
            # this only happens when the column is a storeID, but hopefully in
            # the future we will have more of this.
            tiebreaker = None
        else:
            tiebreaker = self.tableClass.storeID

        tied = lambda a, b: (sortColumn.__get__(a) ==
                             sortColumn.__get__(b))
        def _AND(a, b):
            if a is None:
                return b
            return attributes.AND(a, b)

        results = list(self.store.query(self.tableClass, self.comparison,
                                        sort=sort, limit=pagesize + 1))
        while results:
            if len(results) == 1:
                # XXX TODO: reject 0 pagesize.  If the length of the result set
                # is 1, there's no next result to test for a tie with, so we
                # must be at the end, and we should just yield the result and finish.
                yield results[0]
                return
            for resultidx in range(len(results) - 1):
                # check for a tie.
                result = results[resultidx]
                nextResult = results[resultidx + 1]
                if tied(result, nextResult):
                    # Yield any ties first, in the appropriate order.
                    lastTieBreaker = tiebreaker.__get__(result)
                    # Note that this query is _NOT_ limited: currently large ties
                    # will generate arbitrarily large amounts of work.
                    trq = self.store.query(
                        self.tableClass,
                        _AND(self.comparison,
                             sortColumn == sortColumn.__get__(result)))
                    tiedResults = list(trq)
                    tiedResults.sort(key=lambda rslt: (sortColumn.__get__(result),
                                                       tiebreaker.__get__(result)))
                    for result in tiedResults:
                        yield result
                    # re-start the query here ('result' is set to the
                    # appropriate value by the inner loop)
                    break
                else:
                    yield result

            lastSortValue = sortColumn.__get__(result) # hooray namespace pollution
            results = list(self.store.query(
                    self.tableClass,
                    _AND(self.comparison,
                         sortOp(sortColumn,
                                sortColumn.__get__(result))),
                    sort=sort,
                    limit=pagesize + 1))
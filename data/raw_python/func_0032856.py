def inequalityQuery(self, constraint, count, isAscending):
        """
        Perform a query to obtain some rows from the table represented
        by this model, at the behest of a networked client.

        @param constraint: an additional constraint to apply to the
        query.
        @type constraint: L{axiom.iaxiom.IComparison}.

        @param count: the maximum number of rows to return.
        @type count: C{int}

        @param isAscending: a boolean describing whether the query
        should be yielding ascending or descending results.
        @type isAscending: C{bool}

        @return: an query which will yield some results from this
        model.
        @rtype: L{axiom.iaxiom.IQuery}
        """
        if self.baseConstraint is not None:
            if constraint is not None:
                constraint = AND(self.baseConstraint, constraint)
            else:
                constraint = self.baseConstraint
        # build the sort
        currentSortAttribute = self.currentSortColumn.sortAttribute()
        if isAscending:
            sort = (currentSortAttribute.ascending,
                    self.itemType.storeID.ascending)
        else:
            sort = (currentSortAttribute.descending,
                    self.itemType.storeID.descending)
        return self.store.query(self.itemType, constraint, sort=sort,
                                limit=count).distinct()
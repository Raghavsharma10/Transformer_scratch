def query(self, tableClass, comparison=None,
              limit=None, offset=None, sort=None):
        """
        Return a generator of instances of C{tableClass},
        or tuples of instances if C{tableClass} is a
        tuple of classes.

        Examples::

            fastCars = s.query(Vehicle,
                axiom.attributes.AND(
                    Vehicle.wheels == 4,
                    Vehicle.maxKPH > 200),
                limit=100,
                sort=Vehicle.maxKPH.descending)

            quotesByClient = s.query( (Client, Quote),
                axiom.attributes.AND(
                    Client.active == True,
                    Quote.client == Client.storeID,
                    Quote.created >= someDate),
                limit=10,
                sort=(Client.name.ascending,
                      Quote.created.descending))

        @param tableClass: a subclass of Item to look for instances of,
        or a tuple of subclasses.

        @param comparison: a provider of L{IComparison}, or None, to match
        all items available in the store. If tableClass is a tuple, then
        the comparison must refer to all Item subclasses in that tuple,
        and specify the relationships between them.

        @param limit: an int to limit the total length of the results, or None
        for all available results.

        @param offset: an int to specify a starting point within the available
        results, or None to start at 0.

        @param sort: an L{ISort}, something that comes from an SQLAttribute's
        'ascending' or 'descending' attribute.

        @return: an L{ItemQuery} object, which is an iterable of Items or
        tuples of Items, according to tableClass.
        """
        if isinstance(tableClass, tuple):
            queryClass = MultipleItemQuery
        else:
            queryClass = ItemQuery

        return queryClass(self, tableClass, comparison, limit, offset, sort)
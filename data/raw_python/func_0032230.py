def filterByFilter(self, filterName):
        """
        Swap L{baseConstraint} with the result of calling
        L{IPeopleFilter.getPeopleQueryComparison} on the named filter.

        @type filterName: C{unicode}
        """
        filter = self.filters[filterName]
        self.baseConstraint = filter.getPeopleQueryComparison(self.store)
def getPeopleFilters(self):
        """
        Return an iterator of L{IPeopleFilter} providers available to this
        organizer's store.
        """
        yield AllPeopleFilter()
        yield VIPPeopleFilter()
        for getPeopleFilters in self._gatherPluginMethods('getPeopleFilters'):
            for peopleFilter in getPeopleFilters():
                yield peopleFilter
        for tag in sorted(self.getPeopleTags()):
            yield TaggedPeopleFilter(tag)
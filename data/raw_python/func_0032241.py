def peopleFilters(self, request, tag):
        """
        Return an instance of C{tag}'s I{filter} pattern for each filter we
        get from L{Organizer.getPeopleFilters}, filling the I{name} slot with
        the filter's name.  The first filter will be rendered using the
        I{selected-filter} pattern.
        """
        filters = iter(self.organizer.getPeopleFilters())
        # at some point we might actually want to look at what filter is
        # yielded first, and filter the person list accordingly.  we're just
        # going to assume it's the "All" filter, and leave the person list
        # untouched for now.
        yield tag.onePattern('selected-filter').fillSlots(
            'name', filters.next().filterName)
        pattern = tag.patternGenerator('filter')
        for filter in filters:
            yield pattern.fillSlots('name', filter.filterName)
def by_visits(self, event_kind=None):
        """
        Gets Venues in order of how many Events have been held there.
        Adds a `num_visits` field to each one.

        event_kind filters by kind of Event, e.g. 'theatre', 'cinema', etc.
        """
        qs = self.get_queryset()

        if event_kind is not None:
            qs = qs.filter(event__kind=event_kind)

        qs = qs.annotate(num_visits=Count('event')) \
                .order_by('-num_visits', 'name_sort')

        return qs
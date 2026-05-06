def by_views(self, kind=None):
        """
        Gets Works in order of how many times they've been attached to
        Events.

        kind is the kind of Work, e.g. 'play', 'movie', etc.
        """
        qs = self.get_queryset()

        if kind is not None:
            qs = qs.filter(kind=kind)

        qs = qs.annotate(num_views=Count('event')) \
                .order_by('-num_views', 'title_sort')

        return qs
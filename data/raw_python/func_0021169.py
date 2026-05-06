def by_events(self, kind=None):
        """
        Get the Creators involved in the most Events.

        This only counts Creators directly involved in an Event.
        i.e. if a Creator is the director of a movie Work, and an Event was
        a viewing of that movie, that Event wouldn't count. Unless they were
        also directly involved in the Event (e.g. speaking after the movie).

        kind - If supplied, only Events with that `kind` value will be counted.
        """
        if not spectator_apps.is_enabled('events'):
            raise ImproperlyConfigured("To use the CreatorManager.by_events() method, 'spectator.events' must by in INSTALLED_APPS.")

        qs = self.get_queryset()

        if kind is not None:
            qs = qs.filter(events__kind=kind)

        qs = qs.annotate(num_events=Count('events', distinct=True)) \
                .order_by('-num_events', 'name_sort')

        return qs
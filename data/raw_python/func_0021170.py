def by_works(self, kind=None, role_name=None):
        """
        Get the Creators involved in the most Works.

        kind - If supplied, only Works with that `kind` value will be counted.
        role_name - If supplied, only Works on which the role is that will be counted.

        e.g. To get all 'movie' Works on which the Creators had the role 'Director':

            Creator.objects.by_works(kind='movie', role_name='Director')
        """
        if not spectator_apps.is_enabled('events'):
            raise ImproperlyConfigured("To use the CreatorManager.by_works() method, 'spectator.events' must by in INSTALLED_APPS.")

        qs = self.get_queryset()

        filter_kwargs = {}

        if kind is not None:
            filter_kwargs['works__kind'] = kind

        if role_name is not None:
            filter_kwargs['work_roles__role_name'] = role_name

        if filter_kwargs:
            qs = qs.filter(**filter_kwargs)

        qs = qs.annotate(num_works=Count('works', distinct=True)) \
                .order_by('-num_works', 'name_sort')

        return qs
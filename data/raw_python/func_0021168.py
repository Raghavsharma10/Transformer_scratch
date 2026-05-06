def by_readings(self, role_names=['', 'Author']):
        """
        The Creators who have been most-read, ordered by number of readings.

        By default it will only include Creators whose role was left empty,
        or is 'Author'.

        Each Creator will have a `num_readings` attribute.
        """
        if not spectator_apps.is_enabled('reading'):
            raise ImproperlyConfigured("To use the CreatorManager.by_readings() method, 'spectator.reading' must by in INSTALLED_APPS.")

        qs = self.get_queryset()

        qs = qs.filter(publication_roles__role_name__in=role_names) \
                .exclude(publications__reading__isnull=True) \
                .annotate(num_readings=Count('publications__reading')) \
                .order_by('-num_readings', 'name_sort')

        return qs
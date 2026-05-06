def for_category(self, category, live_only=False):
        """
        Returns queryset of EntryTag instances for specified category.

        :param category: the Category instance.
        :param live_only: flag to include only "live" entries.
        :rtype: django.db.models.query.QuerySet.
        """
        filters = {'tag': category.tag}

        if live_only:
            filters.update({'entry__live': True})

        return self.filter(**filters)
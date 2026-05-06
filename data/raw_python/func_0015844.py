def related_to(self, entry, live_only=False):
        """
        Returns queryset of Entry instances related to specified
        Entry instance.

        :param entry: the Entry instance.
        :param live_only: flag to include only "live" entries.
        :rtype: django.db.models.query.QuerySet.
        """
        filters = {'tag__in': entry.tags}

        if live_only:
            filters.update({'entry__live': True})

        return self.filter(**filters).exclude(entry=entry)
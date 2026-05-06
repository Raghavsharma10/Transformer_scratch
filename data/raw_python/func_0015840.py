def get_queryset(self):
        """
        Returns queryset limited to categories with live Entry instances.

        :rtype: django.db.models.query.QuerySet.
        """
        queryset = super(LiveEntryCategoryManager, self).get_queryset()
        return queryset.filter(tag__in=[
            entry_tag.tag
            for entry_tag
            in EntryTag.objects.filter(entry__live=True)
        ])
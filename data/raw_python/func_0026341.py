def recent(self, check_language=True, language=None, limit=3, exclude=None,
               kwargs=None, category=None):
        """
        Returns recently published new entries.

        """
        if category:
            if not kwargs:
                kwargs = {}
            kwargs['categories__in'] = [category]
        qs = self.published(check_language=check_language, language=language,
                            kwargs=kwargs)
        if exclude:
            qs = qs.exclude(pk=exclude.pk)
        return qs[:limit]
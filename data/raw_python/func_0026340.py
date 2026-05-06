def published(self, check_language=True, language=None, kwargs=None,
                  exclude_kwargs=None):
        """
        Returns all entries, which publication date has been hit or which have
        no date and which language matches the current language.

        """
        if check_language:
            qs = NewsEntry.objects.language(language or get_language()).filter(
                is_published=True)
        else:
            qs = self.get_queryset()
        qs = qs.filter(
            models.Q(pub_date__lte=now()) | models.Q(pub_date__isnull=True)
        )
        if kwargs is not None:
            qs = qs.filter(**kwargs)
        if exclude_kwargs is not None:
            qs = qs.exclude(**exclude_kwargs)
        return qs.distinct().order_by('-pub_date')
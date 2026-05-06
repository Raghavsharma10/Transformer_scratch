def published(self, request=None):
        """
        Returns the published documents in the current language.

        :param request: A Request instance.

        """
        language = getattr(request, 'LANGUAGE_CODE', get_language())
        if not language:
            return self.model.objects.none()

        qs = self.get_queryset()
        qs = qs.filter(
            translations__is_published=True,
            translations__language_code=language,
        )
        # either it has no category or the one it has is published
        qs = qs.filter(
            models.Q(category__isnull=True) |
            models.Q(category__is_published=True))
        return qs
def get_queryset(self):
        """created the base queryset object for the serializer limited to users within the authors
        groups and having `is_staff`

        :return: `django.db.models.QuerySet`
        """
        author_filter = getattr(settings, "BULBS_AUTHOR_FILTER", {"is_staff": True})
        queryset = self.model.objects.filter(**author_filter).distinct()
        return queryset
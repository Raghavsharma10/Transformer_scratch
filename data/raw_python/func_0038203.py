def get_queryset(self):
        """creates the base queryset object for the serializer

        :return: an instance of `django.db.models.QuerySet`
        """
        qs = LogEntry.objects.all()
        content_id = get_query_params(self.request).get("content", None)
        if content_id:
            qs = qs.filter(object_id=content_id)
        return qs
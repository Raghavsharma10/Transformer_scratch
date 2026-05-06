def resolve_objects(cls, objects, skip_cached_urls=False):
        """
        Make sure all AnyUrlValue objects from a set of objects is resolved in bulk.
        This avoids making a query per item.

        :param objects: A list or queryset of models.
        :param skip_cached_urls: Whether to avoid prefetching data that has it's URL cached.
        """
        # Allow the queryset or list to consist of multiple models.
        # This supports querysets from django-polymorphic too.
        queryset = list(objects)
        any_url_values = []

        for obj in queryset:
            model = obj.__class__
            for field in _any_url_fields_by_model[model]:
                any_url_value = getattr(obj, field)
                if any_url_value and any_url_value.url_type.has_id_value:
                    any_url_values.append(any_url_value)

        AnyUrlValue.resolve_values(any_url_values, skip_cached_urls=skip_cached_urls)
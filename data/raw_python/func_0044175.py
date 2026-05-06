def get_queryset(self):
        """
        For reducing the query count the queryset is expanded with `prefetch_related` and `select_related` depending on the
        specified fields and nested fields
        """
        self.queryset = super(CustomFieldsMixin, self).get_queryset()
        serializer_class = self.get_serializer_class()
        if hasattr(serializer_class.Meta, 'nested_fields'):
            nested_fields = serializer_class.Meta.nested_fields
            fields = serializer_class.Meta.fields
            self._expand_queryset(fields, nested_fields, self.queryset.model)
        return self.queryset
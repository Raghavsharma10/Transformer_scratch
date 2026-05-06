def register_queryset(self, queryset, validator=None, default=False):
        """
        Add a given queryset to the iterator with custom logic for iteration.

        :param queryset: List of objects included in the reading list.
        :param validator: Custom logic to determine a queryset's position in a reading_list.
            Validators must accept an index as an argument and return a truthy value.
        :param default: Sets the given queryset as the primary queryset when no validator applies.
        """
        if default or self.default_queryset is None:
            self.default_queryset = queryset
            return
        if validator:
            self.querysets[validator] = queryset
        else:
            raise ValueError(
                """Querysets require validation logic to integrate with reading lists."""
            )
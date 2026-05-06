def or_filter(self, **filters):
        """
        Works like "filter" but joins given filters with OR operator.

        Args:
            **filters: Query filters as keyword arguments.

        Returns:
            Self. Queryset object.

        Example:
            >>> Person.objects.or_filter(age__gte=16, name__startswith='jo')

        """
        clone = copy.deepcopy(self)
        clone.adapter.add_query([("OR_QRY", filters)])
        return clone
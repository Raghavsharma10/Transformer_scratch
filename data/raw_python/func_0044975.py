def exclude(self, **filters):
        """
        Applies query filters for excluding matching records from result set.

        Args:
            **filters: Query filters as keyword arguments.

        Returns:
            Self. Queryset object.

        Examples:
            >>> Person.objects.exclude(age=None)
            >>> Person.objects.filter(name__startswith='jo').exclude(age__lte=16)
        """
        exclude = {'-%s' % key: value for key, value in filters.items()}
        return self.filter(**exclude)
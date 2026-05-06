def order_by(self, *args):
        """
        Applies query ordering.

        Args:
            **args: Order by fields names.
            Defaults to ascending, prepend with hypen (-) for desecending ordering.

        Returns:
            Self. Queryset object.

        Examples:
            >>> Person.objects.order_by('-name', 'join_date')
        """
        clone = copy.deepcopy(self)
        clone.adapter.ordered = True
        if args:
            clone.adapter.order_by(*args)
        return clone
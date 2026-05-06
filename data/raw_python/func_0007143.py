def get_ordering_for_column(self, column, direction):
        """
        Returns a tuple of lookups to order by for the given column
        and direction. Direction is an integer, either -1, 0 or 1.
        """
        if direction == 0:
            return ()
        if column in self.orderings:
            ordering = self.orderings[column]
        else:
            field = self.get_field(column)
            if field is None:
                return ()
            ordering = column
        if not isinstance(ordering, (tuple, list)):
            ordering = [ordering]
        if direction == 1:
            return ordering
        return [lookup[1:] if lookup[0] == '-' else '-' + lookup
                for lookup in ordering]
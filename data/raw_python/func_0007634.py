def serialize(self, value, greedy=True):
        """
        Greedy serialization requires the value to either be a column
        or convertible to a column, whereas non-greedy serialization
        will pass through any string as-is and will only serialize
        Column objects.

        Non-greedy serialization is useful when preparing queries with
        custom filters or segments.
        """

        if greedy and not isinstance(value, Column):
            value = self.normalize(value)

        if isinstance(value, Column):
            return value.id
        else:
            return value
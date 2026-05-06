def from_tuple(cls, query):
        """Create a condition from a query tuple.

        Args:
            query (tuple or list): Tuple or list that contains a query domain
                in the format of ``(field_name, field_value,
                field_value_to)``. ``field_value_to`` is only applicable in
                the case of a date search.

        Returns:
            DomainCondition: An instance of a domain condition. The specific
                type will depend on the data type of the first value provided
                in ``query``.
        """

        field, query = query[0], query[1:]

        try:
            cls = TYPES[type(query[0])]
        except KeyError:
            # We just fallback to the base class if unknown type.
            pass

        return cls(field, *query)
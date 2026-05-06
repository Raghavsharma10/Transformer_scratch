def sorted_fields(cls):
        """Get the names of all writable metadata fields, sorted in the
        order that they should be written.

        This is a lexicographic order, except for instances of
        :class:`DateItemField`, which are sorted in year-month-day
        order.
        """
        for property in sorted(cls.fields(), key=cls._field_sort_name):
            yield property
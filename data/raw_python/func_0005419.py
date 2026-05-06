def _field_sort_name(cls, name):
        """Get a sort key for a field name that determines the order
        fields should be written in.

        Fields names are kept unchanged, unless they are instances of
        :class:`DateItemField`, in which case `year`, `month`, and `day`
        are replaced by `date0`, `date1`, and `date2`, respectively, to
        make them appear in that order.
        """
        if isinstance(cls.__dict__[name], DateItemField):
            name = re.sub('year',  'date0', name)
            name = re.sub('month', 'date1', name)
            name = re.sub('day',   'date2', name)
        return name
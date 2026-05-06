def validate(self, raw_data, **kwargs):
        """The string ``'True'`` (case insensitive) will be converted
        to ``True``, as will any positive integers.

        """
        super(BooleanField, self).validate(raw_data, **kwargs)
        if isinstance(raw_data, string_types):
            valid_data = raw_data.strip().lower() == 'true'
        elif isinstance(raw_data, bool):
            valid_data = raw_data
        else:
            valid_data = raw_data > 0
        return valid_data
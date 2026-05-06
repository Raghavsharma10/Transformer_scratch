def get(cls, name_or_numeric):
        """ Get Enum.Value object matching the value argument.
        :param name_or_numeric: Integer value or attribute name
        :type name_or_numeric: int or str
        :rtype: Enum.Value
        """
        if isinstance(name_or_numeric, six.string_types):
            name_or_numeric = getattr(cls, name_or_numeric.upper())

        return cls.values.get(name_or_numeric)
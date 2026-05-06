def get_field_type(cls, name):
        """
        Takes a field name and gets an appropriate BaseField instance
        for that column.  It inspects the Model that is set on the manager
        to determine what the BaseField subclass should be.

        :param unicode name:
        :return: A BaseField subclass that is appropriate for
            translating a string input into the appropriate format.
        :rtype: ripozo.viewsets.fields.base.BaseField
        """
        python_type = cls._get_field_python_type(cls.model, name)
        if python_type in _COLUMN_FIELD_MAP:
            field_class = _COLUMN_FIELD_MAP[python_type]
            return field_class(name)
        return BaseField(name)
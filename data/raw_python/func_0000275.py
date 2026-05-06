def value_validate(self, value):
        """
        Converts the input single value into the expected Python data type,
        raising django.core.exceptions.ValidationError if the data can't be
        converted.  Returns the converted value. Subclasses should override
        this.
        """
        if not isinstance(value, six.integer_types):
            raise tldap.exceptions.ValidationError("should be a integer")

        try:
            return str(value)
        except (TypeError, ValueError):
            raise tldap.exceptions.ValidationError("is invalid integer")
def value_to_python(self, value):
        """
        Converts the input single value into the expected Python data type,
        raising django.core.exceptions.ValidationError if the data can't be
        converted.  Returns the converted value. Subclasses should override
        this.
        """
        if not isinstance(value, bytes):
            raise tldap.exceptions.ValidationError("should be bytes")
        if value is None:
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            raise tldap.exceptions.ValidationError("is invalid integer")
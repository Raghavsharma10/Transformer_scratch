def value_to_python(self, value):
        """
        Converts the input single value into the expected Python data type,
        raising django.core.exceptions.ValidationError if the data can't be
        converted.  Returns the converted value. Subclasses should override
        this.
        """
        if not isinstance(value, bytes):
            raise tldap.exceptions.ValidationError("should be a bytes")

        try:
            value = int(value)
        except (TypeError, ValueError):
            raise tldap.exceptions.ValidationError("is invalid integer")

        try:
            value = datetime.date.fromtimestamp(value * 24 * 60 * 60)
        except OverflowError:
            raise tldap.exceptions.ValidationError("is too big a date")

        return value
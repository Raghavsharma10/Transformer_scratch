def value_validate(self, value):
        """
        Converts the input single value into the expected Python data type,
        raising django.core.exceptions.ValidationError if the data can't be
        converted.  Returns the converted value. Subclasses should override
        this.
        """
        if not isinstance(value, datetime.date):
            raise tldap.exceptions.ValidationError("is invalid date")
        # a datetime is also a date but they are not compatable
        if isinstance(value, datetime.datetime):
            raise tldap.exceptions.ValidationError("should be a date, not a datetime")
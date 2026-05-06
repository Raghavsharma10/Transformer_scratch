def value_validate(self, value):
        """
        Validates value and throws ValidationError. Subclasses should override
        this to provide validation logic.
        """
        if not isinstance(value, six.string_types):
            raise tldap.exceptions.ValidationError("should be a string")
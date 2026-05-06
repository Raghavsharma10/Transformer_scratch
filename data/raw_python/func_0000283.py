def value_validate(self, value):
        """
        Converts the input single value into the expected Python data type,
        raising django.core.exceptions.ValidationError if the data can't be
        converted.  Returns the converted value. Subclasses should override
        this.
        """
        if not isinstance(value, str):
            raise tldap.exceptions.ValidationError("Invalid sid")

        array = value.split("-")
        length = len(array) - 3

        if length < 1:
            raise tldap.exceptions.ValidationError("Invalid sid")

        if array.pop(0) != "S":
            raise tldap.exceptions.ValidationError("Invalid sid")

        try:
            [int(i) for i in array]
        except TypeError:
            raise tldap.exceptions.ValidationError("Invalid sid")
def to_python(self, value):
        """
        Converts the input value into the expected Python data type, raising
        django.core.exceptions.ValidationError if the data can't be converted.
        Returns the converted value. Subclasses should override this.
        """
        assert isinstance(value, list)

        # convert every value in list
        value = list(value)
        for i, v in enumerate(value):
            value[i] = self.value_to_python(v)

        # return result
        return value
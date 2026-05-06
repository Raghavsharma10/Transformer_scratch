def to_python(self, value):
        """
        Validates that the value is in self.choices and can be coerced to the
        right type.
        """
        value = '' if value in validators.EMPTY_VALUES else smart_text(value)
        if value == self.empty_value or value in validators.EMPTY_VALUES:
            return self.empty_value
        try:
            value = self.coerce(value)
        except (ValueError, TypeError, ValidationError):
            self._on_invalid_value(value)
        return value
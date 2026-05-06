def validate(self, value):
        """
        Validates that the input is in self.choices.
        """
        super(ChoicesField, self).validate(value)
        if value and not self.valid_value(value):
            self._on_invalid_value(value)
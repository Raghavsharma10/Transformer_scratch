def validate(self, value):
        """Validate value."""
        if self.exclusive:
            if value <= self.minimum_value:
                tpl = "'{value}' is lower or equal than minimum ('{min}')."
                raise ValidationError(
                    tpl.format(value=value, min=self.minimum_value))
        else:
            if value < self.minimum_value:
                raise ValidationError(
                    "'{value}' is lower than minimum ('{min}').".format(
                        value=value, min=self.minimum_value))
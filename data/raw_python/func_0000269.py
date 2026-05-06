def clean(self, value):
        """
        Convert the value's type and run validation. Validation errors from
        to_python and validate are propagated. The correct value is returned if
        no error is raised.
        """
        value = self.to_python(value)
        self.validate(value)
        return value
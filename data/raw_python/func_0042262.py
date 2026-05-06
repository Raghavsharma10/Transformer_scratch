def clean(self, value, model_instance):
        """
        Convert the value's type and run validation. Validation errors
        from to_python and validate are propagated. The correct value is
        returned if no error is raised.
        """
        #: return constant's name instead of constant itself
        value = self.to_python(value).name
        self.validate(value, model_instance)
        self.run_validators(value)
        return value
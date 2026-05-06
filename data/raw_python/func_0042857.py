def validate(self, value):
        """Validate a value for this field.  If the field is invalid, this
        will raise a ValueError.  Runs ``pre_validate`` hook prior to
        validation, and returns value if validation passes."""
        value = self.pre_validate(value)
        if not self._typecheck(value):
            raise ValueError('%r failed type check' % value)
        return value
def _verify_validator(self, validator, path):
        """Verifies that a given validator associated with the field at the given path is legitimate."""

        # Validator should be a function
        if not callable(validator):
            raise SchemaFormatException("Invalid validations for {}", path)

        # Validator should accept a single argument
        (args, varargs, keywords, defaults) = getargspec(validator)
        if len(args) != 1:
            raise SchemaFormatException("Invalid validations for {}", path)
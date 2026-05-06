def validate(self, instance):
        """Validates the given document against this schema. Raises a
        ValidationException if there are any failures."""
        errors = {}
        self._validate_instance(instance, errors)

        if len(errors) > 0:
            raise ValidationException(errors)
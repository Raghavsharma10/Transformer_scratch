def validate(self, value):
        """Validate value."""
        flags = self._calculate_flags()

        try:
            result = re.search(self.pattern, value, flags)
        except TypeError as te:
            raise ValidationError(*te.args)

        if not result:
            raise ValidationError(
                'Value "{value}" did not match pattern "{pattern}".'.format(
                    value=value, pattern=self.pattern
                ))
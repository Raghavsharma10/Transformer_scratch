def validate(self, value):
        """Validate value."""
        len_ = len(value)

        if self.minimum_value is not None and len_ < self.minimum_value:
            tpl = "Value '{val}' length is lower than allowed minimum '{min}'."
            raise ValidationError(tpl.format(
                val=value, min=self.minimum_value
            ))

        if self.maximum_value is not None and len_ > self.maximum_value:
            raise ValidationError(
                "Value '{val}' length is bigger than "
                "allowed maximum '{max}'.".format(
                    val=value,
                    max=self.maximum_value,
                ))
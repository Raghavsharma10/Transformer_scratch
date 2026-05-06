def clean(self, value):
        """Take a dirty value and clean it."""
        if (
            self.base_type is not None and
            value is not None and
            not isinstance(value, self.base_type)
        ):
            if isinstance(self.base_type, tuple):
                allowed_types = [typ.__name__ for typ in self.base_type]
                allowed_types_text = ' or '.join(allowed_types)
            else:
                allowed_types_text = self.base_type.__name__
            err_msg = 'Value must be of %s type.' % allowed_types_text
            raise ValidationError(err_msg)

        if not self.has_value(value):
            if self.default is not None:
                raise StopValidation(self.default)

            if self.required:
                raise ValidationError('This field is required.')
            else:
                raise StopValidation(self.blank_value)

        return value
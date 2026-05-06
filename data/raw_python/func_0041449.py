def validate(self, raw_data, **kwargs):
        """Convert the raw_data to an integer.

        """
        try:
            converted_data = int(raw_data)
            return super(IntegerField, self).validate(converted_data)
        except ValueError:
            raise ValidationException(self.messages['invalid'], repr(raw_data))
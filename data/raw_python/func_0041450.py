def validate(self, raw_data, **kwargs):
        """Convert the raw_data to a float.

        """
        try:
            converted_data = float(raw_data)
            super(FloatField, self).validate(converted_data, **kwargs)
            return raw_data
        except ValueError:
            raise ValidationException(self.messages['invalid'], repr(raw_data))
def validate(self, raw_data, **kwargs):
        """The raw_data is returned unchanged."""

        super(DateTimeField, self).validate(raw_data, **kwargs)
        try:
            if isinstance(raw_data, datetime.datetime):
                self.converted = raw_data
            elif self.serial_format is None:
                # parse as iso8601
                self.converted = parse(raw_data)
            else:
                self.converted = datetime.datetime.strptime(raw_data,
                                                            self.serial_format)
            return raw_data
        except (ParseError, ValueError) as e:
            msg = self.messages['parse'] % dict(cls=self.__class__.__name__,
                                                data=raw_data,
                                                format=self.serial_format)
            raise ValidationException(msg, raw_data)
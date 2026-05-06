def parse(self, value):
        """
        Parse date
        """
        value = super(DateOpt, self).parse(value)
        if value is None:
            return None
        if isinstance(value, str):
            value = self.parse_date(value)
        if isinstance(value, datetime) and self.date_only:
            value = value.date()
        return value
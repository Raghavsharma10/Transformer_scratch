def field_date_to_json(self, day):
        """Convert a date to a date triple."""
        if isinstance(day, six.string_types):
            day = parse_date(day)
        return [day.year, day.month, day.day] if day else None
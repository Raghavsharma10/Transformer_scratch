def _validate_iso8601_string(self, value):
        """Return the value or raise a ValueError if it is not a string in ISO8601 format."""
        ISO8601_REGEX = r'(\d{4})-(\d{2})-(\d{2})T(\d{2})\:(\d{2})\:(\d{2})([+-](\d{2})\:(\d{2})|Z)'
        if re.match(ISO8601_REGEX, value):
            return value
        else:
            raise ValueError('{} must be in ISO8601 format.'.format(value))
def parse_date(self, value):
        """A lazy method to parse anything to date.

        If input data type is:

        - string: parse date from it
        - integer: use from ordinal
        - datetime: use date part
        - date: just return it
        """
        if value is None:
            raise Exception("Unable to parse date from %r" % value)
        elif isinstance(value, string_types):
            return self.str2date(value)
        elif isinstance(value, int):
            return date.fromordinal(value)
        elif isinstance(value, datetime):
            return value.date()
        elif isinstance(value, date):
            return value
        else:
            raise Exception("Unable to parse date from %r" % value)
def parse_datetime(self, value):
        """A lazy method to parse anything to datetime.

        If input data type is:

        - string: parse datetime from it
        - integer: use from ordinal
        - date: use date part and set hour, minute, second to zero
        - datetime: just return it
        """
        if value is None:
            raise Exception("Unable to parse datetime from %r" % value)
        elif isinstance(value, string_types):
            return self.str2datetime(value)
        elif isinstance(value, integer_types):
            return self.from_utctimestamp(value)
        elif isinstance(value, float):
            return self.from_utctimestamp(value)
        elif isinstance(value, datetime):
            return value
        elif isinstance(value, date):
            return datetime(value.year, value.month, value.day)
        else:
            raise Exception("Unable to parse datetime from %r" % value)
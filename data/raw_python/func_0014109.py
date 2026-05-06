def from_ISO_8601(cls, date_string, time_string, tz_string):
        """Sufficiently general ISO 8601 parser.

        Inputs must be in "basic" format, i.e. no '-' or ':' separators.
        See https://en.wikipedia.org/wiki/ISO_8601

        """
        # parse tz_string
        if tz_string:
            tz_offset = (int(tz_string[1:3]) * 60) + int(tz_string[3:])
            if tz_string[0] == '-':
                tz_offset = -tz_offset
        else:
            tz_offset = None
        if time_string == '000000':
            # assume no time information
            time_string = ''
            tz_offset = None
        datetime_string = date_string + time_string[:13]
        precision = min((len(datetime_string) - 2) // 2, 7)
        if precision <= 0:
            return None
        fmt = ''.join(('%Y', '%m', '%d', '%H', '%M', '%S', '.%f')[:precision])
        return cls(
            (datetime.strptime(datetime_string, fmt), precision, tz_offset))
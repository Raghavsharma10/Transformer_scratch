def field_datetime_to_json(self, dt):
        """Convert a datetime to a UTC timestamp w/ microsecond resolution.

        datetimes w/o timezone will be assumed to be in UTC
        """
        if isinstance(dt, six.string_types):
            dt = parse_datetime(dt)
        if not dt:
            return None
        ts = timegm(dt.utctimetuple())
        if dt.microsecond:
            return "{0}.{1:0>6d}".format(ts, dt.microsecond)
        else:
            return ts
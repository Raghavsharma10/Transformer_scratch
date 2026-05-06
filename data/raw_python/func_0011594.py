def field_timedelta_to_json(self, td):
        """Convert timedelta to value containing total number of seconds.

        If there are fractions of a second the return value will be a
        string, otherwise it will be an int.
        """
        if isinstance(td, six.string_types):
            td = parse_duration(td)
        if not td:
            return None
        if td.microseconds > 0:
            return str(td.total_seconds())
        else:
            return int(td.total_seconds())
def from_iso_permissive(datestring, use_dateutil=True):
        """Parse an ISO8601-formatted datetime and return a datetime object.

        Inspired by the marshmallow.utils.from_iso function, but also accepts
        datestrings that don't contain the time.
        """
        dateutil_available = False
        try:
            from dateutil import parser
            dateutil_available = True
        except ImportError:
            dateutil_available = False
            import datetime

        # Use dateutil's parser if possible
        if dateutil_available and use_dateutil:
            return parser.parse(datestring)
        else:
            # Strip off timezone info.
            return datetime.datetime.strptime(datestring[:19],
                                              '%Y-%m-%dT%H:%M:%S')
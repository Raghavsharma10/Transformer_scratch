def match_timestamp(self, start_time, end_time, match):
        """Matches the time of this log entry.

        arg:    start_time (osid.calendaring.DateTime): start time
        arg:    end_time (osid.calendaring.DateTime): end time
        arg:    match (boolean): ``true`` if for a positive match,
                ``false`` for a negative match
        raise:  InvalidArgument - ``start_time`` is greater than
                ``end_time``
        raise:  NullArgument - ``start_time`` or ``end_time`` is
                ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        self._match_minimum_date_time('timestamp', start_time, match)
        self._match_maximum_date_time('timestamp', end_time, match)
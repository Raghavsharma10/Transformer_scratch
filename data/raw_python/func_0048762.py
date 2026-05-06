def match_start_time(self, start, end, match):
        """Matches assessments whose start time falls between the specified range inclusive.

        arg:    start (osid.calendaring.DateTime): start of range
        arg:    end (osid.calendaring.DateTime): end of range
        arg:    match (boolean): ``true`` for a positive match,
                ``false`` for a negative match
        raise:  InvalidArgument - ``end`` is less than ``start``
        *compliance: mandatory -- This method must be implemented.*

        """
        self._match_minimum_date_time('startTime', start, match)
        self._match_maximum_date_time('startTime', end, match)